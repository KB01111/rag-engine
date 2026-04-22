package main

import (
	"testing"
)

func TestTruncate_ShortString(t *testing.T) {
	input := "hello"
	got := truncate(input, 10)
	if got != input {
		t.Errorf("truncate(%q, 10) = %q; want %q", input, got, input)
	}
}

func TestTruncate_ExactLength(t *testing.T) {
	input := "hello"
	got := truncate(input, 5)
	if got != input {
		t.Errorf("truncate(%q, 5) = %q; want %q", input, got, input)
	}
}

func TestTruncate_LongString(t *testing.T) {
	input := "hello world"
	got := truncate(input, 5)
	want := "hello..."
	if got != want {
		t.Errorf("truncate(%q, 5) = %q; want %q", input, got, want)
	}
}

func TestTruncate_EmptyString(t *testing.T) {
	got := truncate("", 5)
	if got != "" {
		t.Errorf("truncate(%q, 5) = %q; want empty string", "", got)
	}
}

func TestTruncate_ZeroMaxLen(t *testing.T) {
	// When maxLen=0, s[:0] is empty, so the result is the ellipsis alone.
	input := "hello"
	got := truncate(input, 0)
	want := "..."
	if got != want {
		t.Errorf("truncate(%q, 0) = %q; want %q", input, got, want)
	}
}

func TestTruncate_Unicode(t *testing.T) {
	// truncate operates on bytes, so verify it doesn't panic on multi-byte chars
	input := "αβγδε"
	got := truncate(input, 4)
	// 4 bytes cuts into the second character (each Greek letter is 2 bytes in UTF-8)
	if len(got) == 0 {
		t.Errorf("truncate on unicode string returned empty")
	}
}

func TestBoolPtr_True(t *testing.T) {
	p := boolPtr(true)
	if p == nil {
		t.Fatal("boolPtr(true) returned nil")
	}
	if *p != true {
		t.Errorf("*boolPtr(true) = %v; want true", *p)
	}
}

func TestBoolPtr_False(t *testing.T) {
	p := boolPtr(false)
	if p == nil {
		t.Fatal("boolPtr(false) returned nil")
	}
	if *p != false {
		t.Errorf("*boolPtr(false) = %v; want false", *p)
	}
}

func TestBoolPtr_IsPointer(t *testing.T) {
	// Verify that two separate calls return distinct pointers.
	p1 := boolPtr(true)
	p2 := boolPtr(true)
	if p1 == p2 {
		t.Errorf("boolPtr should return a new pointer each call; got same address")
	}
}

func TestBoolPtr_Mutation(t *testing.T) {
	// Mutating the returned pointer should not affect subsequent calls.
	p := boolPtr(true)
	*p = false
	p2 := boolPtr(true)
	if !*p2 {
		t.Errorf("mutating a previously returned pointer affected a new call")
	}
}