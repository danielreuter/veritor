//! The hash framing of `veritor/protocol/merkle.py`, bit for bit.
//!
//! ```text
//! _hash(tag, *parts) = SHA-256( FRAME || u32be(len(tag)) || tag
//!                               || ( u64be(len(part)) || part )* )
//! _uint(v)           = minimal big-endian bytes of v, at least one byte
//! domain_id          = _hash(b"domain", binding, _uint(owner + 2), identity, _uint(count))
//! leaf               = _hash(b"leaf", domain_id, _uint(rank), _uint(position), schema, value)
//! node               = _hash(b"node", domain_id, _uint(level), _uint(index), left, right)
//! empty_root         = _hash(b"empty", domain_id)
//! ```
//!
//! and the tagged hash of `veritor/core/identity.py`:
//!
//! ```text
//! tagged_sha256(tag, payload) = SHA-256( TAGGED_PREFIX || u32be(len(tag)) || tag
//!                                        || u64be(len(payload)) || payload )
//! ```

use sha2::{Digest, Sha256};

/// `merkle._FRAME`.
pub const FRAME: &[u8] = b"veritor/protocol/merkle/frame/v3\0";
/// `identity._TAGGED_HASH_PREFIX`.
pub const TAGGED_PREFIX: &[u8] = b"veritor/tagged-sha256/v1\0";

pub const LEAF_TAG: &[u8] = b"leaf";
pub const NODE_TAG: &[u8] = b"node";
pub const EMPTY_TAG: &[u8] = b"empty";
pub const DOMAIN_TAG: &[u8] = b"domain";

/// Plain SHA-256 of `data` (the statement digest in the public values).
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// `merkle._uint`: minimal big-endian encoding, at least one byte.
pub fn uint(value: u64) -> [u8; 8] {
    value.to_be_bytes()
}

/// The number of significant bytes of `uint(value)`: `max(1, ceil(bit_length / 8))`.
pub fn uint_len(value: u64) -> usize {
    let bits = 64 - value.leading_zeros() as usize;
    core::cmp::max(1, bits.div_ceil(8))
}

fn uint_slice(value: u64, buffer: &[u8; 8]) -> &[u8] {
    &buffer[8 - uint_len(value)..]
}

/// Frames up to this many bytes are assembled on the stack and hashed in one
/// `update` (a leaf or node frame is ~180 bytes); larger ones fall back to
/// streaming.  One `update` call is markedly cheaper in the zkVM than ten.
const STACK_FRAME: usize = 512;

/// `merkle._hash(tag, *parts)`.
pub fn framed_hash(tag: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let total = FRAME.len() + 4 + tag.len() + parts.iter().map(|part| 8 + part.len()).sum::<usize>();
    if total <= STACK_FRAME {
        let mut buffer = [0u8; STACK_FRAME];
        let mut at = 0;
        let mut put = |bytes: &[u8]| {
            buffer[at..at + bytes.len()].copy_from_slice(bytes);
            at += bytes.len();
        };
        put(FRAME);
        put(&(tag.len() as u32).to_be_bytes());
        put(tag);
        for part in parts {
            put(&(part.len() as u64).to_be_bytes());
            put(part);
        }
        return sha256(&buffer[..total]);
    }
    let mut hasher = Sha256::new();
    hasher.update(FRAME);
    hasher.update((tag.len() as u32).to_be_bytes());
    hasher.update(tag);
    for part in parts {
        hasher.update((part.len() as u64).to_be_bytes());
        hasher.update(part);
    }
    hasher.finalize().into()
}

/// `identity.tagged_sha256(tag, payload)`.
pub fn tagged_sha256(tag: &[u8], payload: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(TAGGED_PREFIX);
    hasher.update((tag.len() as u32).to_be_bytes());
    hasher.update(tag);
    hasher.update((payload.len() as u64).to_be_bytes());
    hasher.update(payload);
    hasher.finalize().into()
}

/// `CommitmentDomain.domain_id`; `owner_plus_two` is `owner + 2` (weights 0, boundary 1, interior r + 2).
pub fn domain_id(
    binding: &[u8; 32],
    owner_plus_two: u64,
    identity_digest: &[u8; 32],
    count: u64,
) -> [u8; 32] {
    let owner = uint(owner_plus_two);
    let count_bytes = uint(count);
    framed_hash(
        DOMAIN_TAG,
        &[
            binding,
            uint_slice(owner_plus_two, &owner),
            identity_digest,
            uint_slice(count, &count_bytes),
        ],
    )
}

/// `CommitmentDomain.leaf(rank, position, schema, value)`.
pub fn leaf(domain: &[u8; 32], rank: u64, position: u64, schema: &[u8], value: &[u8]) -> [u8; 32] {
    let rank_bytes = uint(rank);
    let position_bytes = uint(position);
    framed_hash(
        LEAF_TAG,
        &[
            domain,
            uint_slice(rank, &rank_bytes),
            uint_slice(position, &position_bytes),
            schema,
            value,
        ],
    )
}

/// `CommitmentDomain.node(level, index, left, right)`.
pub fn node(domain: &[u8; 32], level: u64, index: u64, left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let level_bytes = uint(level);
    let index_bytes = uint(index);
    framed_hash(
        NODE_TAG,
        &[
            domain,
            uint_slice(level, &level_bytes),
            uint_slice(index, &index_bytes),
            left,
            right,
        ],
    )
}

/// `CommitmentDomain.empty_root()`.
pub fn empty_root(domain: &[u8; 32]) -> [u8; 32] {
    framed_hash(EMPTY_TAG, &[domain])
}

/// `merkle.merkle_depth(count)`: `0` for at most one leaf, else `bit_length(count - 1)`.
pub fn merkle_depth(count: u64) -> usize {
    if count <= 1 {
        0
    } else {
        64 - (count - 1).leading_zeros() as usize
    }
}

/// Fold an authentication path from a leaf at `rank` up to the root.
pub fn fold_path(domain: &[u8; 32], rank: u64, leaf_digest: [u8; 32], path: &[[u8; 32]]) -> [u8; 32] {
    let mut digest = leaf_digest;
    let mut cursor = rank;
    for (level, sibling) in path.iter().enumerate() {
        digest = if cursor % 2 == 0 {
            node(domain, level as u64, cursor >> 1, &digest, sibling)
        } else {
            node(domain, level as u64, cursor >> 1, sibling, &digest)
        };
        cursor >>= 1;
    }
    digest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uint_is_minimal_and_at_least_one_byte() {
        assert_eq!(uint_len(0), 1);
        assert_eq!(uint_len(1), 1);
        assert_eq!(uint_len(255), 1);
        assert_eq!(uint_len(256), 2);
        assert_eq!(uint_len(u64::MAX), 8);
        let zero = uint(0);
        assert_eq!(uint_slice(0, &zero), &[0u8]);
        let big = uint(0x0102_03);
        assert_eq!(uint_slice(0x0102_03, &big), &[1u8, 2, 3]);
    }

    #[test]
    fn depth_matches_python() {
        assert_eq!(merkle_depth(0), 0);
        assert_eq!(merkle_depth(1), 0);
        assert_eq!(merkle_depth(2), 1);
        assert_eq!(merkle_depth(3), 2);
        assert_eq!(merkle_depth(4), 2);
        assert_eq!(merkle_depth(5), 3);
        assert_eq!(merkle_depth(1 << 20), 20);
        assert_eq!(merkle_depth((1 << 20) + 1), 21);
    }
}
