use crate::core::packet::{EncodedDelta, EncodedValues, SparseDelta};
use anyhow::{Result, bail, ensure};

/// How delta values are encoded on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValueEncoding {
    /// Exact 4-byte floats
    #[default]
    F32,
    /// 1 byte per value, scaled by the largest absolute value
    Int8,
    /// 1 bit per value (its sign), scaled by the mean absolute value
    Sign,
}

impl ValueEncoding {
    /// Parses "f32", "int8" or "sign".
    ///
    /// # Errors
    ///
    /// Returns an error for any other name.
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "f32" => Ok(Self::F32),
            "int8" => Ok(Self::Int8),
            "sign" => Ok(Self::Sign),
            other => bail!("Unknown value encoding '{other}', expected f32, int8 or sign"),
        }
    }

    /// The name accepted by parse.
    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Int8 => "int8",
            Self::Sign => "sign",
        }
    }
}

/// Encodes a sparse delta for the wire.
///
/// Lossy encodings can round values to zero; those entries are dropped.
/// The decoded result is what receivers apply, so senders must update their
/// anchor with decode_delta(encode_delta(..)), not with the original values.
///
/// # Arguments
///
/// * `delta` - Delta with strictly increasing indices and finite values
/// * `encoding` - How to encode the values
pub fn encode_delta(delta: &SparseDelta, encoding: ValueEncoding) -> EncodedDelta {
    let (indices, values): (Vec<u32>, EncodedValues) = match encoding {
        ValueEncoding::F32 => (
            delta.indices.clone(),
            EncodedValues::F32(delta.values.clone()),
        ),
        ValueEncoding::Int8 => {
            let max = delta.values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let scale = max / 127.0;
            let mut indices = Vec::with_capacity(delta.indices.len());
            let mut data = Vec::with_capacity(delta.values.len());
            if scale > 0.0 {
                for (&idx, &val) in delta.indices.iter().zip(&delta.values) {
                    let q = (val / scale).round().clamp(-127.0, 127.0) as i8;
                    if q != 0 {
                        indices.push(idx);
                        data.push(q);
                    }
                }
            }
            (indices, EncodedValues::Int8 { scale, data })
        }
        ValueEncoding::Sign => {
            let nonzero: Vec<(u32, f32)> = delta
                .indices
                .iter()
                .zip(&delta.values)
                .filter(|(_, v)| **v != 0.0)
                .map(|(&i, &v)| (i, v))
                .collect();
            let scale = if nonzero.is_empty() {
                0.0
            } else {
                nonzero.iter().map(|(_, v)| v.abs()).sum::<f32>() / nonzero.len() as f32
            };
            let mut negative = vec![0u8; nonzero.len().div_ceil(8)];
            for (i, (_, v)) in nonzero.iter().enumerate() {
                if *v < 0.0 {
                    negative[i / 8] |= 1 << (i % 8);
                }
            }
            let indices = nonzero.into_iter().map(|(i, _)| i).collect();
            (indices, EncodedValues::Sign { scale, negative })
        }
    };

    EncodedDelta {
        sequence_id: delta.sequence_id,
        count: indices.len() as u32,
        index_gaps: encode_index_gaps(&indices),
        values,
    }
}

/// Decodes a wire delta back to indices and values.
///
/// # Errors
///
/// Returns an error if the gaps are truncated or overflow, or if the number
/// of indices or values does not match `count`.
pub fn decode_delta(encoded: &EncodedDelta) -> Result<SparseDelta> {
    let count = encoded.count as usize;
    let indices = decode_index_gaps(&encoded.index_gaps)?;
    ensure!(
        indices.len() == count,
        "Delta has {} indices, expected {count}",
        indices.len()
    );

    let values: Vec<f32> = match &encoded.values {
        EncodedValues::F32(values) => values.clone(),
        EncodedValues::Int8 { scale, data } => data.iter().map(|&q| q as f32 * scale).collect(),
        EncodedValues::Sign { scale, negative } => {
            ensure!(
                negative.len() == count.div_ceil(8),
                "Sign bitmap has {} bytes, expected {}",
                negative.len(),
                count.div_ceil(8)
            );
            (0..count)
                .map(|i| {
                    if negative[i / 8] & (1 << (i % 8)) != 0 {
                        -scale
                    } else {
                        *scale
                    }
                })
                .collect()
        }
    };
    ensure!(
        values.len() == count,
        "Delta has {} values, expected {count}",
        values.len()
    );

    Ok(SparseDelta {
        sequence_id: encoded.sequence_id,
        indices,
        values,
    })
}

/// Encodes strictly increasing indices as LEB128 gaps.
pub fn encode_index_gaps(indices: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(indices.len() + indices.len() / 4);
    let mut prev = 0u32;
    for (i, &idx) in indices.iter().enumerate() {
        let gap = if i == 0 { idx } else { idx - prev };
        write_varint(&mut out, gap);
        prev = idx;
    }
    out
}

/// Decodes LEB128 gaps back to indices.
///
/// # Errors
///
/// Returns an error if a varint is truncated or an index exceeds u32.
pub fn decode_index_gaps(bytes: &[u8]) -> Result<Vec<u32>> {
    let mut indices = Vec::new();
    let mut pos = 0;
    let mut prev: Option<u32> = None;
    while pos < bytes.len() {
        let gap = read_varint(bytes, &mut pos)?;
        let idx = match prev {
            None => gap,
            Some(p) => {
                ensure!(gap > 0, "Repeated index after {p}");
                p.checked_add(gap)
                    .ok_or_else(|| anyhow::anyhow!("Index overflows u32"))?
            }
        };
        indices.push(idx);
        prev = Some(idx);
    }
    Ok(indices)
}

/// Appends a u32 as LEB128 (7 bits per byte, high bit = more bytes follow).
fn write_varint(out: &mut Vec<u8>, mut value: u32) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// Reads one LEB128 u32 starting at `pos`, advancing it.
fn read_varint(bytes: &[u8], pos: &mut usize) -> Result<u32> {
    let mut value: u64 = 0;
    for shift in (0..35).step_by(7) {
        let Some(&byte) = bytes.get(*pos) else {
            bail!("Truncated varint");
        };
        *pos += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return u32::try_from(value).map_err(|_| anyhow::anyhow!("Varint overflows u32"));
        }
    }
    bail!("Varint longer than 5 bytes")
}

#[cfg(test)]
mod test;
