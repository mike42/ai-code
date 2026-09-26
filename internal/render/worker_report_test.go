package render

import "testing"

func TestWorkerReportLabelIsPickedOut(t *testing.T) {
	text := "<worker-report id=\"2\" label=\"trace the callers\" turns=\"4\">\nfindings\n</worker-report>"
	got, ok := workerReportLabel(text)
	if !ok {
		t.Fatal("a worker report was not recognised, so its body would be printed twice")
	}
	if got != "trace the callers" {
		t.Errorf("label = %q, want %q", got, "trace the callers")
	}
}

// Anything else is shown in full, not summarised to nothing.
func TestAnOrdinarySteerIsNotTreatedAsAReport(t *testing.T) {
	for _, in := range []string{
		"stop and check the tests first",
		"<worker-report id=\"1\">no label here</worker-report>",
		"",
	} {
		if _, ok := workerReportLabel(in); ok {
			t.Errorf("%q was taken for a worker report", in)
		}
	}
}
