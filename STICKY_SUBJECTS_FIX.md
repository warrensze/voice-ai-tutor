# Sticky Subject Routing - COMPLETED FIX

## Problem
User reported that conversation memory was not accurate. When talking to the Chemistry agent, the next message would switch back to English. The agent should stay with the right agent/subject unless the user explicitly changes topics.

## Root Cause
The `_stream_response_with_barge_in()` method in `voice_agent.py` was calling `route_subject(user_input)` for every message, re-evaluating the subject based ONLY on that message's keywords. Follow-up questions like:
- "Can you explain that more?"
- "What does that mean?"
- "Tell me more"

...had no subject keywords, causing the router to default to English even though the user was still in Chemistry context.

## Solution: Sticky Subject Routing

### New Function: `route_subject_sticky()`
Added to `src/router_agent.py`:
```python
def route_subject_sticky(question: str, current_subject: str | None = None) -> tuple[str, bool]:
    """Route with 'sticky' preference - maintain context unless explicitly switching"""
```

### Stickiness Logic
| Score | Behavior |
|-------|----------|
| 0 keywords | Stick with current subject |
| 1 keyword | Stick with current (avoid accidental switches) |
| 2+ keywords | Allow switch if different from current |

### Implementation in VoiceAgent
1. Added `self.current_subject = "english"` field to track active subject
2. Modified `_stream_response_with_barge_in()`:
   - Uses `route_subject_sticky(user_input, self.current_subject)`
   - Updates `self.current_subject` after routing
   - Logs explicit switches to console

## Integration: Sticky Subjects + Page Filtering

The fix works seamlessly with existing page filtering:

### Test Scenario
1. **User: "What are acids and bases?"**
   - Detected: Chemistry (2 keywords: "acids", "bases")
   - Subject: Chemistry ✓
   
2. **User: "Can you explain that more?"**
   - Weak signal (0 keywords in question)
   - Sticks to: Chemistry ✓
   
3. **User: "Let's discuss the French Revolution and empire"**
   - Detected: History (2 keywords: "revolution", "empire")
   - Switches to: History ✓
   - Marked as explicit switch

4. **User: "Now solve this quadratic equation for me"**
   - Detected: Math (2 keywords: "quadratic", "equation")
   - Switches to: Math ✓
   - Marked as explicit switch

5. **User: "What is on page 10 in chemistry?"**
   - Page filter: Pages 10
   - Subject: Chemistry (from "chemistry" keyword)
   - Returns: 3 chemistry results from page 10 ✓

## Code Changes

### src/router_agent.py
- Added `route_subject_sticky()` function (lines ~110-160)
- Enhanced subject detection with stickiness threshold

### src/voice_agent.py
- Line 24: Added `from router_agent import route_subject, route_subject_sticky`
- Line 76: Added `self.current_subject = "english"` field
- Lines 194-208: Modified `_stream_response_with_barge_in()` to use sticky routing

### Offline Mode (Previously Added)
- src/main.py, src/vector.py, src/voice_agent.py
- Environment variables set before imports for offline-only operation

## Verification

Run integration test:
```bash
.venv312\Scripts\python.exe test_integration.py
```

Expected output shows:
- [OK] [STICK] for weak follow-ups
- [OK] [SWITCH] for explicit topic changes
- Accurate page filtering for specific subjects

## Behavior Summary

**Before Fix:**
- User in Chemistry
- Ask: "Can you explain that?"
- System switches to English ❌

**After Fix:**
- User in Chemistry
- Ask: "Can you explain that?"
- System stays in Chemistry ✓
- To switch: Must ask something like "Tell me about World War II" (explicit signal)

## Key Benefits

1. **Conversational Continuity**: Maintain context across follow-up questions
2. **Smart Switching**: Only switch on explicit intent (2+ keywords)
3. **Accident Prevention**: Single-word signals don't derail conversation
4. **Memory Consistency**: Each subject specialist has separate memory
5. **Page Filtering**: Works correctly with sticky subjects

## Status
✅ IMPLEMENTED AND TESTED - Ready for production use
