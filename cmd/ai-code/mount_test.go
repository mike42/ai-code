package main

import "testing"

// The model is told where it is, and inside a container that is not where it
// was launched from. Every path it derives is wrong otherwise.
func TestContainerPathsReplaceHostPaths(t *testing.T) {
	cases := []struct{ project, cwd, folder, want string }{
		{"/home/dev/proj", "/home/dev/proj", "/workspace", "/workspace"},
		{"/home/dev/proj", "/home/dev/proj/examples/demo", "/workspace", "/workspace/examples/demo"},
		{"/home/dev/proj", "/somewhere/else", "/workspace", "/workspace"},
		{"/home/dev/proj", "/home/dev/proj/src", "/workspaces/proj", "/workspaces/proj/src"},
		// workspaceFolder is the configuration's to choose; an empty one falls
		// back to /workspace.
		{"/home/dev/proj", "/home/dev/proj/src", "", "/workspace/src"},
	}
	for _, c := range cases {
		if got := mountCwd(c.project, c.cwd, c.folder); got != c.want {
			t.Errorf("mountCwd(%q, %q, %q) = %q, want %q", c.project, c.cwd, c.folder, got, c.want)
		}
	}
}
