package agent

import (
	"fmt"
	"strings"

	"ai-code/internal/provider"
)

// Validate checks the message list against the invariants the chat-completions
// contract enforces. It runs before every request, so a malformed list left by
// a cancellation is named while the cause is still in view.
func Validate(messages []provider.Message) error {
	var problems []string

	for i, m := range messages {
		switch m.Role {
		case provider.RoleAssistant:
			if len(m.ToolCalls) == 0 {
				continue
			}
			// Every tool call must be answered, in the messages that immediately follow.
			answered := map[string]bool{}
			for j := i + 1; j < len(messages); j++ {
				if messages[j].Role != provider.RoleTool {
					break
				}
				answered[messages[j].ToolCallID] = true
			}
			for _, tc := range m.ToolCalls {
				if tc.ID == "" {
					problems = append(problems, fmt.Sprintf(
						"message %d: assistant tool call %q has no id", i, tc.Name))
					continue
				}
				if !answered[tc.ID] {
					problems = append(problems, fmt.Sprintf(
						"message %d: tool call %s (%s) has no matching tool result. "+
							"Every tool call must be answered before the next request",
						i, tc.ID, tc.Name))
				}
			}

		case provider.RoleTool:
			if m.ToolCallID == "" {
				problems = append(problems, fmt.Sprintf(
					"message %d: tool result has no tool_call_id", i))
				continue
			}
			// A tool result must follow the assistant message that requested it.
			if !precededByCall(messages, i, m.ToolCallID) {
				problems = append(problems, fmt.Sprintf(
					"message %d: tool result %s answers a tool call that is not immediately above it",
					i, m.ToolCallID))
			}
		}
	}

	if len(problems) > 0 {
		return fmt.Errorf("conversation is malformed and was not sent:\n  %s",
			strings.Join(problems, "\n  "))
	}
	return nil
}

func precededByCall(messages []provider.Message, idx int, callID string) bool {
	for j := idx - 1; j >= 0; j-- {
		switch messages[j].Role {
		case provider.RoleTool:
			continue
		case provider.RoleAssistant:
			for _, tc := range messages[j].ToolCalls {
				if tc.ID == callID {
					return true
				}
			}
			return false
		default:
			return false
		}
	}
	return false
}

// Repair rewrites a conversation into one the provider will accept. The
// contract is total: Validate must pass on whatever comes out.
//
// The repairs, in the order applied to each assistant message:
//
//   - A tool call with no id is dropped: it cannot be answered.
//   - A call with an id but no result gets a synthesised result saying it may
//     have partially run.
//   - A tool result answering nothing immediately above it is dropped.
//   - An assistant message left with no calls, no content and no reasoning is
//     dropped.
func Repair(messages []provider.Message) ([]provider.Message, int) {
	out := make([]provider.Message, 0, len(messages))
	repaired := 0

	for i := 0; i < len(messages); i++ {
		m := messages[i]

		// A tool result reached here is not inside a call's answer run, so it
		// answers nothing.
		if m.Role == provider.RoleTool {
			repaired++
			continue
		}
		if m.Role != provider.RoleAssistant || len(m.ToolCalls) == 0 {
			out = append(out, m)
			continue
		}

		// The full-slice expression forces the append below to allocate rather
		// than write through.
		kept := m.ToolCalls[:0:0]
		for _, tc := range m.ToolCalls {
			if tc.ID == "" {
				repaired++
				continue
			}
			kept = append(kept, tc)
		}
		m.ToolCalls = kept

		// Consume the run of tool results that follows this message.
		results := map[string]provider.Message{}
		extra := 0
		j := i + 1
		for ; j < len(messages) && messages[j].Role == provider.RoleTool; j++ {
			r := messages[j]
			if _, dup := results[r.ToolCallID]; r.ToolCallID == "" || dup {
				extra++
				continue
			}
			results[r.ToolCallID] = r
		}
		i = j - 1
		repaired += extra

		if len(m.ToolCalls) == 0 {
			// Every call was a fragment; keep the message only if it also said
			// something.
			if strings.TrimSpace(m.Content) != "" || strings.TrimSpace(m.Reasoning) != "" {
				out = append(out, m)
			} else {
				repaired++
			}
			repaired += len(results)
			continue
		}

		out = append(out, m)
		for _, tc := range m.ToolCalls {
			if r, ok := results[tc.ID]; ok {
				out = append(out, r)
				delete(results, tc.ID)
				continue
			}
			out = append(out, provider.Message{
				Role:       provider.RoleTool,
				ToolCallID: tc.ID,
				Name:       tc.Name,
				IsError:    true,
				Content: "Interrupted before this tool reported a result. " +
					"It may have run partially, so any state it touches may have been modified. " +
					"Re-check rather than assuming it did nothing.",
			})
			repaired++
		}
		// Anything still here answered a call that is not in this message.
		repaired += len(results)
	}

	return out, repaired
}
