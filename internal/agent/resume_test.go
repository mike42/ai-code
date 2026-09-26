package agent

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"ai-code/internal/provider"
	"ai-code/internal/tool"
)

// resumeAgent has a many-turn session in a wide window, so a plan for a narrow
// one has real trimming to describe.
func resumeAgent(t *testing.T, turns int) (*Agent, *scriptedClient) {
	t.Helper()
	client := &scriptedClient{}
	a := New(client, "m", toolExecutor(t), &collectSink{}, Options{ContextLimit: 262144})

	body := strings.Repeat("the quick brown fox jumps over the lazy dog. ", 40)
	var msgs []provider.Message
	for i := range turns {
		msgs = append(msgs,
			provider.Message{Role: provider.RoleUser, Content: fmt.Sprintf("ask %d: %s", i, body)},
			provider.Message{Role: provider.RoleAssistant, Content: fmt.Sprintf("say %d: %s", i, body)})
	}
	a.SetMessages(msgs)
	return a, client
}

func TestPlanResumeCallsNoModel(t *testing.T) {
	a, client := resumeAgent(t, 40)
	a.SetSummary("## Goal\nship it", 20)

	for _, limit := range []int{0, 4096, 16384, 65536, 1 << 20} {
		a.PlanResume(limit)
	}
	if n := len(client.requests()); n != 0 {
		t.Errorf("planning made %d requests, want none", n)
	}
}

func TestPlanResumeFitsWhenThereIsRoomToAnswer(t *testing.T) {
	a, _ := resumeAgent(t, 2)

	if p := a.PlanResume(262144); !p.Fits {
		t.Errorf("a short session in a wide window should fit: %+v", p)
	}
	// Doubled: the reserve is clamped to half a window this small, so half of
	// what is asked for is what the session gets.
	tight := 2 * (a.TranscriptTokens() + minOutputTokens/2)
	if room := a.usableIn(tight) - a.TranscriptTokens(); room >= minOutputTokens {
		t.Fatalf("the fixture leaves %d tokens spare, which is not tight", room)
	}
	if p := a.PlanResume(tight); p.Fits {
		t.Errorf("a window with under %d tokens spare should not count as fitting: %+v",
			minOutputTokens, p)
	}
	if p := a.PlanResume(0); !p.Fits {
		t.Error("an unknown window has nothing to decide, so nothing should be reported")
	}
}

// The checkpoint costs room and covers the past; the transcript keeps more
// recent messages and covers nothing.
func TestPlanResumeSeparatesTheTwoWays(t *testing.T) {
	a, _ := resumeAgent(t, 60)
	a.SetSummary("## Goal\nship it\n\n## Next Steps\n1. keep going", 40)

	p := a.PlanResume(32768)
	if p.Fits {
		t.Fatal("a 60-turn session should not fit a 32k window")
	}
	if !p.Checkpoint.Available {
		t.Fatal("a checkpoint exists, so it should be offered")
	}
	if p.Checkpoint.Kept >= p.Transcript.Kept {
		t.Errorf("the checkpoint costs room, so it should keep fewer messages: %d vs %d",
			p.Checkpoint.Kept, p.Transcript.Kept)
	}
	if p.Transcript.Unrepresented != p.Transcript.Dropped {
		t.Errorf("without a checkpoint nothing speaks for the dropped messages: %d of %d",
			p.Transcript.Unrepresented, p.Transcript.Dropped)
	}
	if p.Checkpoint.Unrepresented >= p.Checkpoint.Dropped {
		t.Errorf("the checkpoint covers %d messages, so it should speak for some of the "+
			"%d dropped ones, not %d", 40, p.Checkpoint.Dropped, p.Checkpoint.Unrepresented)
	}
	for _, plan := range []ResumePlan{p.Checkpoint, p.Transcript} {
		if plan.Prompt <= 0 || plan.Prompt > p.Usable {
			t.Errorf("a plan sends %d tokens of a %d budget: %+v", plan.Prompt, p.Usable, plan)
		}
		if plan.Free != p.Usable-plan.Prompt {
			t.Errorf("free room %d does not account for %d sent of %d",
				plan.Free, plan.Prompt, p.Usable)
		}
	}
}

func TestPlanResumeWithoutACheckpoint(t *testing.T) {
	a, _ := resumeAgent(t, 60)

	p := a.PlanResume(32768)
	if p.Checkpoint.Available {
		t.Error("no checkpoint has been written, so none should be offered")
	}
	if p.Transcript.Kept == 0 {
		t.Error("the transcript plan has to carry something")
	}
}

// The transcript choice has to change what goes on the wire, or it is a label.
func TestResumeFromTranscriptDropsTheCheckpoint(t *testing.T) {
	a, _ := resumeAgent(t, 60)
	a.SetSummary("## Goal\nship it", 40)

	if !containsCheckpoint(a.MessagesFitting(6000)) {
		t.Fatal("by default a narrow window should carry the checkpoint")
	}
	a.SetResumeFromTranscript(true)
	if containsCheckpoint(a.MessagesFitting(6000)) {
		t.Error("the transcript choice should stop the checkpoint going out")
	}
}

func TestModelChangeClearsTheTranscriptPreference(t *testing.T) {
	a, _ := resumeAgent(t, 60)
	a.SetSummary("## Goal\nship it", 40)
	a.SetResumeFromTranscript(true)

	a.SetModel("other", 262144, 0)
	if a.resumeFromTranscript {
		t.Error("a model change should clear the choice made about the previous window")
	}
}

func TestTranscriptChoiceSuppressesAutoSummarising(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{{text: "carrying on"}}}
	exec := tool.NewLocalExecutor(tool.NewState(t.TempDir()))
	a := New(client, "m", exec, &collectSink{}, Options{
		ContextLimit: 16384, AutoCompact: true,
	})
	fillSession(t, a, 9000)
	a.SetResumeFromTranscript(true)

	if err := a.Run(context.Background(), "go on"); err != nil {
		t.Fatal(err)
	}
	// One request: the turn itself. A second would be an unasked-for summary.
	if n := len(client.requests()); n != 1 {
		t.Errorf("%d requests, want 1 -- the extra one is a summary nobody asked for", n)
	}
	if s, _ := a.Summary(); s != "" {
		t.Errorf("a checkpoint was written anyway: %q", s)
	}
}

func TestStoppedMidTurn(t *testing.T) {
	call := provider.Message{Role: provider.RoleAssistant, ToolCalls: []provider.ToolCall{
		{ID: "1", Name: "read", Args: `{"path":"x"}`}}}
	result := provider.Message{Role: provider.RoleTool, ToolCallID: "1", Content: "contents"}

	for _, tc := range []struct {
		name string
		msgs []provider.Message
		want bool
	}{
		{"empty session", nil, false},
		{"answered", []provider.Message{
			{Role: provider.RoleUser, Content: "hi"},
			{Role: provider.RoleAssistant, Content: "done"}}, false},
		{"asked but not answered", []provider.Message{
			{Role: provider.RoleUser, Content: "hi"}}, true},
		{"tool call with no result", []provider.Message{
			{Role: provider.RoleUser, Content: "hi"}, call}, true},
		{"results the model never read back", []provider.Message{
			{Role: provider.RoleUser, Content: "hi"}, call, result}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, _ := resumeAgent(t, 0)
			a.SetMessages(tc.msgs)
			if got := a.StoppedMidTurn(); got != tc.want {
				t.Errorf("StoppedMidTurn = %v, want %v", got, tc.want)
			}
		})
	}
}

func containsCheckpoint(msgs []provider.Message) bool {
	for _, m := range msgs {
		if strings.Contains(m.Content, "<context-checkpoint>") {
			return true
		}
	}
	return false
}

func TestCheckpointWorthwhileIsFalseOnAColdSession(t *testing.T) {
	client := &scriptedClient{}
	a := New(client, "m", toolExecutor(t), &collectSink{}, Options{ContextLimit: 262144})

	if a.CheckpointWorthwhile(SpeculativeSummaryMaxTokens) {
		t.Error("an empty session has nothing to checkpoint")
	}
	a.SetMessages([]provider.Message{{Role: provider.RoleUser, Content: "hello"}})
	if a.CheckpointWorthwhile(SpeculativeSummaryMaxTokens) {
		t.Error("a single message has nothing to checkpoint")
	}
}

func TestCheckpointWorthwhileIsFalseWhenNothingHasHappenedSince(t *testing.T) {
	a, _ := resumeAgent(t, 40)

	if !a.CheckpointWorthwhile(SpeculativeSummaryMaxTokens) {
		t.Fatal("a session with no checkpoint at all is worth checkpointing")
	}

	// Stand in for a checkpoint covering exactly what one written now would.
	a.SetSummary("## Goal\nship it", a.checkpointThrough(SpeculativeSummaryMaxTokens))

	if a.CheckpointWorthwhile(SpeculativeSummaryMaxTokens) {
		t.Error("checkpointing again with no new conversation buys nothing and costs a model call")
	}
}

func TestCheckpointWorthwhileReturnsWhenTheSessionGrows(t *testing.T) {
	a, _ := resumeAgent(t, 40)
	a.SetSummary("## Goal\nship it", a.checkpointThrough(SpeculativeSummaryMaxTokens))

	grown, _ := resumeAgent(t, 120)
	a.SetMessages(grown.Messages())

	if !a.CheckpointWorthwhile(SpeculativeSummaryMaxTokens) {
		t.Error("a session that has grown past the old cut is worth checkpointing again")
	}
}
