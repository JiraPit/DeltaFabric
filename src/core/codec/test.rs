#[cfg(test)]
mod codec_tests {
    use crate::core::codec::{
        ValueEncoding, decode_delta, decode_index_gaps, encode_delta, encode_index_gaps,
    };
    use crate::core::packet::{DeltaPacket, EncodedValues, SparseDelta, to_bytes};

    fn delta(indices: Vec<u32>, values: Vec<f32>) -> SparseDelta {
        SparseDelta {
            sequence_id: 7,
            indices,
            values,
        }
    }

    /// Deterministic pseudo-random values in [-1, 1].
    fn pseudo_random(seed: u64, len: usize) -> Vec<f32> {
        let mut x = seed;
        (0..len)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn norm(values: &[f32]) -> f32 {
        values.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    #[test]
    fn test_parse_and_name() {
        for encoding in [ValueEncoding::F32, ValueEncoding::Int8, ValueEncoding::Sign] {
            assert_eq!(ValueEncoding::parse(encoding.name()).unwrap(), encoding);
        }
        assert!(ValueEncoding::parse("f16").is_err());
        assert_eq!(ValueEncoding::default(), ValueEncoding::F32);
    }

    #[test]
    fn test_index_gaps_roundtrip_edge_values() {
        let indices = vec![0, 1, 127, 128, 255, 16_383, 16_384, 2_097_151, u32::MAX];
        let bytes = encode_index_gaps(&indices);

        assert_eq!(decode_index_gaps(&bytes).unwrap(), indices);
    }

    #[test]
    fn test_index_gaps_use_one_byte_for_small_gaps() {
        let indices: Vec<u32> = (0..100).map(|i| i * 100).collect();
        assert_eq!(encode_index_gaps(&indices).len(), 100);

        assert_eq!(encode_index_gaps(&[128]), vec![0x80, 0x01]);
        assert!(encode_index_gaps(&[]).is_empty());
    }

    #[test]
    fn test_index_gaps_reject_malformed() {
        // Truncated: continuation bit set on the last byte.
        assert!(decode_index_gaps(&[0x80]).is_err());
        // Too long for u32.
        assert!(decode_index_gaps(&[0xff, 0xff, 0xff, 0xff, 0xff, 0x01]).is_err());
        // Value above u32::MAX.
        assert!(decode_index_gaps(&[0xff, 0xff, 0xff, 0xff, 0x1f]).is_err());
        // Repeated index (zero gap after the first).
        assert!(decode_index_gaps(&[5, 0]).is_err());
        // Sum overflows u32.
        let mut bytes = encode_index_gaps(&[u32::MAX]);
        bytes.push(1);
        assert!(decode_index_gaps(&bytes).is_err());
    }

    #[test]
    fn test_f32_roundtrip_is_exact() {
        let original = delta(vec![3, 10, 4000], vec![0.25, -1.5, 1e-7]);
        let encoded = encode_delta(&original, ValueEncoding::F32);

        assert_eq!(encoded.count, 3);
        assert_eq!(decode_delta(&encoded).unwrap(), original);
    }

    #[test]
    fn test_int8_roundtrip_error_is_bounded() {
        let values = pseudo_random(1, 200);
        let original = delta((0..200).map(|i| i * 3).collect(), values.clone());
        let max = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));

        let decoded = decode_delta(&encode_delta(&original, ValueEncoding::Int8)).unwrap();

        for (&idx, &val) in decoded.indices.iter().zip(&decoded.values) {
            let orig = values[(idx / 3) as usize];
            assert!((orig - val).abs() <= max / 254.0 + 1e-6);
        }
        assert!(decoded.values.iter().any(|v| (v.abs() - max).abs() < 1e-6));
    }

    #[test]
    fn test_int8_drops_values_rounding_to_zero() {
        let original = delta(vec![1, 2, 3], vec![1.0, 0.001, -0.5]);
        let encoded = encode_delta(&original, ValueEncoding::Int8);
        let decoded = decode_delta(&encoded).unwrap();

        assert_eq!(encoded.count, 2);
        assert_eq!(decoded.indices, vec![1, 3]);
        assert_eq!(decoded.values[0], 1.0);
        assert!((decoded.values[1] + 0.5).abs() < 0.005);
    }

    #[test]
    fn test_sign_uses_mean_absolute_value() {
        let original = delta(vec![0, 5, 9], vec![3.0, -1.0, 2.0]);
        let decoded = decode_delta(&encode_delta(&original, ValueEncoding::Sign)).unwrap();

        assert_eq!(decoded.indices, vec![0, 5, 9]);
        assert_eq!(decoded.values, vec![2.0, -2.0, 2.0]);
    }

    #[test]
    fn test_sign_always_shrinks_the_error() {
        // Mean scaling makes sign compression a contraction: what is left to
        // send after a packet is always smaller than before.
        for seed in 0..20 {
            let values = pseudo_random(seed, 37);
            let original = delta((0..37).collect(), values.clone());
            let decoded = decode_delta(&encode_delta(&original, ValueEncoding::Sign)).unwrap();

            let residual: Vec<f32> = values
                .iter()
                .zip(&decoded.values)
                .map(|(v, q)| v - q)
                .collect();
            assert!(norm(&residual) < norm(&values));
        }
    }

    #[test]
    fn test_sign_bitmap_crosses_byte_boundaries() {
        let values: Vec<f32> = (0..17)
            .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        let original = delta((0..17).collect(), values.clone());
        let encoded = encode_delta(&original, ValueEncoding::Sign);

        match &encoded.values {
            EncodedValues::Sign { negative, .. } => assert_eq!(negative.len(), 3),
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(decode_delta(&encoded).unwrap().values, values);
    }

    #[test]
    fn test_empty_delta_encodes_to_nothing() {
        for encoding in [ValueEncoding::F32, ValueEncoding::Int8, ValueEncoding::Sign] {
            let encoded = encode_delta(&delta(vec![], vec![]), encoding);
            assert_eq!(encoded.count, 0);
            assert!(decode_delta(&encoded).unwrap().indices.is_empty());
        }
    }

    #[test]
    fn test_decode_rejects_count_mismatches() {
        let original = delta(vec![1, 2, 3], vec![1.0, -2.0, 3.0]);
        for encoding in [ValueEncoding::F32, ValueEncoding::Int8, ValueEncoding::Sign] {
            let mut encoded = encode_delta(&original, encoding);
            encoded.count = 4;
            assert!(decode_delta(&encoded).is_err(), "{encoding:?}");
        }

        let mut short_values = encode_delta(&original, ValueEncoding::F32);
        short_values.values = EncodedValues::F32(vec![1.0]);
        assert!(decode_delta(&short_values).is_err());
    }

    #[test]
    fn test_packet_sizes_shrink_with_encoding() {
        // 1% of a 40k-weight model: ~400 entries about 100 apart.
        let indices: Vec<u32> = (0..400).map(|i| i * 100 + (i % 7)).collect();
        let original = delta(indices, pseudo_random(3, 400));
        let size = |encoding| {
            let packet = DeltaPacket {
                delta: encode_delta(&original, encoding),
                init_hash: None,
            };
            to_bytes(&packet).unwrap().len()
        };

        let f32_size = size(ValueEncoding::F32);
        let int8_size = size(ValueEncoding::Int8);
        let sign_size = size(ValueEncoding::Sign);

        // Previously 8 bytes per entry (u32 index + f32 value).
        assert!(f32_size < 400 * 5 + 64, "f32 {f32_size}");
        assert!(int8_size < 400 * 2 + 64, "int8 {int8_size}");
        assert!(sign_size < 400 + 400 / 8 + 64, "sign {sign_size}");
        assert!(sign_size < int8_size && int8_size < f32_size);
    }
}
