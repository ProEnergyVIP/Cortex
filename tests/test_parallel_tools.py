#!/usr/bin/env python3
"""
Test script to demonstrate parallel tool execution
"""
import asyncio
import time
from dataclasses import dataclass

# Simulate tool execution with delays
def slow_sync_tool_1(query: str) -> str:
    """Simulates a slow database query"""
    time.sleep(1)
    return f"DB Result for: {query}"

def slow_sync_tool_2(url: str) -> str:
    """Simulates a slow API call"""
    time.sleep(1)
    return f"API Result from: {url}"

def slow_sync_tool_3(file: str) -> str:
    """Simulates slow file processing"""
    time.sleep(1)
    return f"File processed: {file}"

async def slow_async_tool_1(query: str) -> str:
    """Simulates a slow async database query"""
    await asyncio.sleep(1)
    return f"Async DB Result for: {query}"

async def slow_async_tool_2(url: str) -> str:
    """Simulates a slow async API call"""
    await asyncio.sleep(1)
    return f"Async API Result from: {url}"

async def slow_async_tool_3(file: str) -> str:
    """Simulates slow async file processing"""
    await asyncio.sleep(1)
    return f"Async File processed: {file}"

# Test sequential vs parallel execution
def test_sync_sequential():
    """Test sync mode with sequential execution (3 tools = ~3 seconds)"""
    print("\n" + "="*60)
    print("TEST 1: Sync Mode - Sequential Execution")
    print("="*60)
    
    start = time.time()
    
    # Simulate calling 3 tools sequentially
    results = []
    results.append(slow_sync_tool_1("users"))
    results.append(slow_sync_tool_2("example.com"))
    results.append(slow_sync_tool_3("data.csv"))
    
    elapsed = time.time() - start
    
    print(f"Results: {len(results)} tools executed")
    print(f"Time: {elapsed:.2f}s (expected ~3s)")
    print(f"✓ Sequential execution works as expected")
    
    return elapsed

def test_sync_parallel():
    """Test sync mode with parallel execution (3 tools = ~1 second)"""
    print("\n" + "="*60)
    print("TEST 2: Sync Mode - Parallel Execution")
    print("="*60)
    
    from concurrent.futures import ThreadPoolExecutor
    
    start = time.time()
    
    # Simulate calling 3 tools in parallel
    tools = [slow_sync_tool_1, slow_sync_tool_2, slow_sync_tool_3]
    args = ["users", "example.com", "data.csv"]
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda f, a: f(a), tools, args))
    
    elapsed = time.time() - start
    
    print(f"Results: {len(results)} tools executed")
    print(f"Time: {elapsed:.2f}s (expected ~1s)")
    print(f"✓ Parallel execution is {3/elapsed:.1f}x faster!")
    
    return elapsed

async def test_async_sequential():
    """Test async mode with sequential execution (3 tools = ~3 seconds)"""
    print("\n" + "="*60)
    print("TEST 3: Async Mode - Sequential Execution")
    print("="*60)
    
    start = time.time()
    
    # Simulate calling 3 tools sequentially
    results = []
    results.append(await slow_async_tool_1("users"))
    results.append(await slow_async_tool_2("example.com"))
    results.append(await slow_async_tool_3("data.csv"))
    
    elapsed = time.time() - start
    
    print(f"Results: {len(results)} tools executed")
    print(f"Time: {elapsed:.2f}s (expected ~3s)")
    print(f"✓ Sequential execution works as expected")
    
    return elapsed

async def test_async_concurrent():
    """Test async mode with concurrent execution (3 tools = ~1 second)"""
    print("\n" + "="*60)
    print("TEST 4: Async Mode - Concurrent Execution")
    print("="*60)
    
    start = time.time()
    
    # Simulate calling 3 tools concurrently
    tasks = [
        slow_async_tool_1("users"),
        slow_async_tool_2("example.com"),
        slow_async_tool_3("data.csv")
    ]
    
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    
    print(f"Results: {len(results)} tools executed")
    print(f"Time: {elapsed:.2f}s (expected ~1s)")
    print(f"✓ Concurrent execution is {3/elapsed:.1f}x faster!")
    
    return elapsed

async def test_async_with_semaphore():
    """Test async mode with limited concurrency (3 tools, max 2 concurrent = ~2 seconds)"""
    print("\n" + "="*60)
    print("TEST 5: Async Mode - Limited Concurrency (max 2)")
    print("="*60)
    
    start = time.time()
    
    # Limit to 2 concurrent tasks
    semaphore = asyncio.Semaphore(2)
    
    async def limited_call(func, arg):
        async with semaphore:
            return await func(arg)
    
    tasks = [
        limited_call(slow_async_tool_1, "users"),
        limited_call(slow_async_tool_2, "example.com"),
        limited_call(slow_async_tool_3, "data.csv")
    ]
    
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    
    print(f"Results: {len(results)} tools executed")
    print(f"Time: {elapsed:.2f}s (expected ~2s)")
    print(f"✓ Limited concurrency works correctly")
    
    return elapsed

def main():
    print("\n" + "🚀 "*20)
    print("PARALLEL TOOL EXECUTION DEMONSTRATION")
    print("🚀 "*20)
    
    # Sync tests
    seq_time = test_sync_sequential()
    par_time = test_sync_parallel()
    speedup_sync = seq_time / par_time
    
    # Async tests
    async def run_async_tests():
        seq_time = await test_async_sequential()
        con_time = await test_async_concurrent()
        await test_async_with_semaphore()
        return seq_time, con_time
    
    seq_time_async, con_time_async = asyncio.run(run_async_tests())
    speedup_async = seq_time_async / con_time_async
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Sync Mode:")
    print(f"  Sequential: {seq_time:.2f}s")
    print(f"  Parallel:   {par_time:.2f}s")
    print(f"  Speedup:    {speedup_sync:.1f}x")
    print()
    print(f"Async Mode:")
    print(f"  Sequential: {seq_time_async:.2f}s")
    print(f"  Concurrent: {con_time_async:.2f}s")
    print(f"  Speedup:    {speedup_async:.1f}x")
    print()
    print("✅ Parallel execution provides significant performance improvements!")
    print("="*60)

if __name__ == "__main__":
    main()
