"""Unit tests for all-SWA RadixCache release semantics."""

import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocation import retire_window
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.srt.mem_cache.pure_swa_radix_cache import PureSWARadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _FakeReqToTokenPool:
    def __init__(self, req_to_token):
        self.req_to_token = req_to_token

    def write(self, indices, values):
        self.req_to_token[indices] = values


class _FakeAllocator:
    """Rows routed to the full side are skipped: all-SWA has no full pool."""

    device = "cpu"
    storage = None
    # The pools take their rows from memories of their own.
    pools_share_memory = False

    def window_pool_names(self):
        return ("swa",)

    def window_available_size(self, sub_pool):
        return self.swa_available_size()

    def target_window_pool(self):
        return "swa"

    def get_kvcache(self):
        return SimpleNamespace(request_window=None)

    def __init__(self, page_size=1):
        self.page_size = page_size
        self.freed = []
        self.skipped = []
        self.window_freed = []

    def free_segment(self, indices, *, start_pos):
        self.freed.extend(indices.tolist())

    def free_segments(self, segments):
        for indices, start_pos in segments:
            self.free_segment(indices, start_pos=start_pos)

    def free_full_segments(self, segments):
        for indices, _ in segments:
            self.skipped.extend(indices.tolist())

    def free_swa_segment(self, indices, *, start_pos):
        self.window_freed.extend(indices.tolist())


def _make_cache(*, disable, num_tokens, page_size=1, eviction_policy="lru"):
    req_to_token_pool = _FakeReqToTokenPool(
        torch.arange(num_tokens, dtype=torch.int64).unsqueeze(0)
    )
    allocator = _FakeAllocator(page_size)
    # The one window pool keeps the cache's 4-token window, as the cache
    # builder binds it.
    kv_storage_of(req_to_token_pool, allocator).bind_window_retention(4, None)
    return PureSWARadixCache(
        CacheInitParams(
            disable=disable,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            sliding_window_size=4,
            eviction_policy=eviction_policy,
        )
    )


class TestPureSWARadixCache(CustomTestCase):
    def test_finish_inserts_up_to_the_evict_floor_and_frees_the_rest(self):
        cache = _make_cache(disable=False, num_tokens=8)
        allocator = cache.token_to_kv_pool_allocator
        token_ids = array("q", range(8))
        req = SimpleNamespace(
            origin_input_ids=token_ids,
            output_ids=array("q"),
            full_untruncated_fill_ids=token_ids,
            extra_key=None,
            cache_salt=None,
            last_node=cache.root_node,
            lock=None,
            priority=0,
            kv=ReqKvInfo(
                req_pool_idx=0,
                window_evict_floor=4,
                window_evicted_seqlens={"swa": 6},
            ),
        )

        cache.checkpoint(req, up_to=8)
        cache.free_kv_row(req.kv, [(req.kv.cache_protected_len, 8)])
        cache.unlock(req.lock)

        # [0, 4) went into the tree; [4, 6) was window-evicted; [6, 8) is freed.
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
        self.assertEqual(match.device_prefix_len, 4)
        self.assertEqual(allocator.freed, [6, 7])
        self.assertEqual(allocator.skipped, [4, 5])


class TestDisabledPureSWARadixCache(CustomTestCase):
    def test_finished_req_skips_protected_prefix_and_evicted_range(self):
        cache = _make_cache(disable=True, num_tokens=10)
        allocator = cache.token_to_kv_pool_allocator
        token_ids = array("q", range(8))
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
        req = SimpleNamespace(
            origin_input_ids=token_ids,
            output_ids=array("q"),
            full_untruncated_fill_ids=token_ids,
            extra_key=None,
            cache_salt=None,
            last_node=match.last_device_node,
            lock=cache.lock(match.last_device_node),
            priority=0,
            kv=ReqKvInfo(
                req_pool_idx=0,
                cache_protected_len=2,
                window_evict_floor=3,
                window_evicted_seqlens={"swa": 6},
            ),
        )

        # protected 2, floor 3, cursor 6: [2, 3) and [6, 8) go back, [3, 6) is dead
        cache.checkpoint(req, up_to=8)
        cache.free_kv_row(req.kv, [(req.kv.cache_protected_len, 8)])
        cache.unlock(req.lock)

        self.assertEqual(req.kv.cache_protected_len, 2)
        self.assertEqual(allocator.freed, [2, 6, 7])
        self.assertEqual(allocator.skipped, [3, 4, 5])
        self.assertEqual(cache.total_size(), 0)

    def test_window_eviction_frees_up_to_the_window(self):
        cache = _make_cache(disable=True, num_tokens=20, page_size=8)
        req = SimpleNamespace(kv=ReqKvInfo(req_pool_idx=0))

        retire_window(req, 20, cache)

        # pre_len 20 - window 4 = 16, page-aligned: [0, 16) slides out. A
        # prefix-sharing cache would keep max(window, page) and stop at 8.
        self.assertEqual(cache.token_to_kv_pool_allocator.window_freed, list(range(16)))
        self.assertEqual(req.kv.get_evicted_seqlen("swa"), 16)


class TestPureSWADecodeReclaim(CustomTestCase):
    def test_window_demand_evicts_from_the_single_pool(self):
        """A window demand (what a decode preallocation reclaims for) evicts
        from the one pool an all-SWA cache keeps, though the tree has no
        window slot."""
        cache = _make_cache(disable=False, num_tokens=16)
        evicted = []

        def evict(params):
            evicted.append(params)
            return EvictResult()

        cache.evict = evict
        allocator = cache.token_to_kv_pool_allocator
        allocator.swa_available_size = lambda: 2
        allocator.full_available_size = lambda: 2
        cache.evict_for_alloc(KVDemand(window_tokens={"swa": 6}))

        self.assertEqual(len(evicted), 1)
        self.assertEqual(evicted[0].num_tokens, 4)


class TestRadixSplitCreationTime(CustomTestCase):
    def test_fifo_eviction_preserves_creation_time_across_node_split(self):
        """Splitting an older node must keep its original creation_time so FIFO
        evicts the split prefix before a branch inserted later."""
        clock = iter(float(i) for i in range(1, 100))
        with mock.patch(
            "sglang.srt.mem_cache.radix_cache.time.monotonic", side_effect=clock
        ):
            cache = _make_cache(disable=False, num_tokens=16, eviction_policy="fifo")
            branch_a = RadixKey(array("q", [1, 2, 3, 4]))
            branch_b = RadixKey(array("q", [10, 11, 12, 13]))
            cache.insert(
                InsertParams(key=branch_a, value=torch.arange(4, dtype=torch.int64))
            )
            cache.insert(
                InsertParams(key=branch_b, value=torch.arange(4, 8, dtype=torch.int64))
            )

            # Split branch_a into prefix [1, 2] and suffix [3, 4] after branch_b exists.
            split_match = cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", [1, 2])))
            )
            self.assertEqual(split_match.device_prefix_len, 2)

            # Evicting 4 tokens must drop branch_a's suffix [3, 4] and then its
            # prefix [1, 2], leaving the newer branch_b cached.
            evict_result = cache.evict(EvictParams(num_tokens=4))
            self.assertEqual(evict_result.num_tokens_evicted, 4)

            self.assertEqual(
                cache.match_prefix(MatchPrefixParams(key=branch_a)).device_prefix_len, 0
            )
            self.assertEqual(
                cache.match_prefix(MatchPrefixParams(key=branch_b)).device_prefix_len, 4
            )

    def test_filo_eviction_preserves_creation_time_across_node_split(self):
        """Splitting an older node must not make its prefix look newer than a
        later-inserted branch under FILO eviction."""
        clock = iter(float(i) for i in range(1, 100))
        with mock.patch(
            "sglang.srt.mem_cache.radix_cache.time.monotonic", side_effect=clock
        ):
            cache = _make_cache(disable=False, num_tokens=16, eviction_policy="filo")
            branch_a = RadixKey(array("q", [1, 2, 3, 4]))
            branch_b = RadixKey(array("q", [10, 11, 12, 13]))
            cache.insert(
                InsertParams(key=branch_a, value=torch.arange(4, dtype=torch.int64))
            )
            res_b = cache.insert(
                InsertParams(key=branch_b, value=torch.arange(4, 8, dtype=torch.int64))
            )

            # Split branch_a after branch_b was inserted.
            cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", [1, 2]))))

            # Evict branch_a's suffix while branch_b is locked so prefix [1, 2]
            # becomes an evictable leaf alongside branch_b.
            lock_b = cache.lock(res_b.last_device_node)
            cache.evict(EvictParams(num_tokens=2))
            cache.unlock(lock_b)

            # FILO must evict the newer branch_b before branch_a's older prefix.
            cache.evict(EvictParams(num_tokens=2))
            self.assertEqual(
                cache.match_prefix(MatchPrefixParams(key=branch_b)).device_prefix_len, 0
            )
            self.assertEqual(
                cache.match_prefix(MatchPrefixParams(key=branch_a)).device_prefix_len, 2
            )


if __name__ == "__main__":
    unittest.main()
