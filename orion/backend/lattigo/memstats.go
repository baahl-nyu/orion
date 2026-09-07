package main

import "C"

import (
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// Working-set measurement for the CPU backend.
//
// Peak RSS of a Go process measures the allocator (how much garbage the
// collector let pile up before running), not the computation: the same
// cell varies by several GiB between runs. What corresponds to the GPU
// backend's counter (peak of live allocated bytes) is the peak of the live
// heap, i.e. HeapAlloc read right after a forced collection, sampled
// through the run. The sampler runs a collection every interval and
// records the largest post-collection HeapAlloc it saw. Collections are
// cheap here: the heap is a few large no-scan slices, so a cycle is
// milliseconds even at tens of GiB. It slows the run, so use it for memory
// cells only, never timed ones.

var (
	sampMu   sync.Mutex
	sampStop chan struct{}
	sampDone chan struct{}
	sampPeak uint64
)

func liveHeapBytes() uint64 {
	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.HeapAlloc
}

//export LiveHeapBytes
func LiveHeapBytes() C.ulonglong {
	b := liveHeapBytes()
	for {
		p := atomic.LoadUint64(&sampPeak)
		if b <= p || atomic.CompareAndSwapUint64(&sampPeak, p, b) {
			break
		}
	}
	return C.ulonglong(b)
}

//export StartLiveHeapSampler
func StartLiveHeapSampler(intervalMs C.int) {
	sampMu.Lock()
	defer sampMu.Unlock()
	if sampStop != nil {
		return
	}
	atomic.StoreUint64(&sampPeak, 0)
	sampStop = make(chan struct{})
	sampDone = make(chan struct{})
	interval := time.Duration(intervalMs) * time.Millisecond
	go func(stop, done chan struct{}) {
		defer close(done)
		for {
			select {
			case <-stop:
				return
			case <-time.After(interval):
				LiveHeapBytes()
			}
		}
	}(sampStop, sampDone)
}

//export StopLiveHeapSampler
func StopLiveHeapSampler() C.ulonglong {
	sampMu.Lock()
	defer sampMu.Unlock()
	if sampStop != nil {
		close(sampStop)
		<-sampDone
		sampStop, sampDone = nil, nil
	}
	LiveHeapBytes()
	return C.ulonglong(atomic.LoadUint64(&sampPeak))
}
