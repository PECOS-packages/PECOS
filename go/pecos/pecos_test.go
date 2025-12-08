// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
// the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package pecos

import (
	"strings"
	"testing"
)

func TestVersion(t *testing.T) {
	version := Version()
	if !strings.Contains(version, "PECOS") {
		t.Errorf("Version should contain 'PECOS', got: %s", version)
	}
	if !strings.Contains(version, "Go") {
		t.Errorf("Version should contain 'Go', got: %s", version)
	}
}

func TestNewQubitId(t *testing.T) {
	tests := []struct {
		name    string
		index   int64
		wantNil bool
		wantIdx int64
	}{
		{"zero index", 0, false, 0},
		{"positive index", 42, false, 42},
		{"large index", 1000000, false, 1000000},
		{"negative index", -1, true, 0},
		{"negative index -100", -100, true, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			q := NewQubitId(tt.index)
			if tt.wantNil {
				if q != nil {
					t.Errorf("NewQubitId(%d) = %v, want nil", tt.index, q)
				}
			} else {
				if q == nil {
					t.Fatalf("NewQubitId(%d) = nil, want non-nil", tt.index)
				}
				if q.Index() != tt.wantIdx {
					t.Errorf("QubitId.Index() = %d, want %d", q.Index(), tt.wantIdx)
				}
			}
		})
	}
}

func TestQubitIdString(t *testing.T) {
	q := NewQubitId(5)
	if q == nil {
		t.Fatal("NewQubitId(5) returned nil")
	}

	str := q.String()
	if !strings.Contains(str, "QubitId") {
		t.Errorf("QubitId.String() should contain 'QubitId', got: %s", str)
	}
	if !strings.Contains(str, "5") {
		t.Errorf("QubitId.String() should contain '5', got: %s", str)
	}
}

func TestAddTwoNumbers(t *testing.T) {
	tests := []struct {
		a, b, want int64
	}{
		{1, 2, 3},
		{0, 0, 0},
		{-5, 10, 5},
		{100, -100, 0},
		{1000000, 2000000, 3000000},
	}

	for _, tt := range tests {
		got := AddTwoNumbers(tt.a, tt.b)
		if got != tt.want {
			t.Errorf("AddTwoNumbers(%d, %d) = %d, want %d", tt.a, tt.b, got, tt.want)
		}
	}
}
