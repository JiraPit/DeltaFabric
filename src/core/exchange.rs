use crate::core::networking::Session;
use crate::core::packet::{decode_packet, decode_snapshot};
use crate::core::sync::SyncState;
use anyhow::Result;
use tracing::warn;

/// Runs the network side of one training step.
///
/// Answers snapshot requests from resyncing peers, applies received deltas
/// to the mixing sum, completes a running resync with arrived snapshots and
/// requests the snapshots still missing. Never touches the model weights, so
/// it can run every step without extracting parameters.
///
/// # Arguments
///
/// * `session` - The node's network session
/// * `state` - The node's sync state (should be initialized)
///
/// # Errors
///
/// Returns an error if replying to or requesting a snapshot fails.
pub async fn exchange(session: &mut Session, state: &mut SyncState) -> Result<()> {
    // Requests are dropped until there is an anchor to share; the requester retries.
    let requests = session.pull_snapshot_requests();
    if state.is_initialized() {
        let snapshot = state.snapshot();
        for query in requests {
            session.reply_snapshot(query, &snapshot).await?;
        }
    }

    for (origin, payload) in session.pull_packets() {
        match decode_packet(&payload) {
            Ok(packet) => {
                state.receive(origin, packet);
            }
            Err(e) => warn!(origin = %origin, error = %e, "Failed to decode delta packet"),
        }
    }

    for (origin, payload) in session.poll_snapshots() {
        let result = decode_snapshot(&payload).and_then(|s| state.receive_snapshot(origin, s));
        if let Err(e) = result {
            warn!(origin = %origin, error = %e, "Rejected anchor snapshot");
        }
    }

    for peer in state.missing_snapshots() {
        session.request_snapshot(peer).await?;
    }

    Ok(())
}
