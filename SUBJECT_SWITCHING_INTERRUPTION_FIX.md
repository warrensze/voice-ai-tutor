# Subject Switching & Interruption Fix - Summary

## Issues Reported
1. **Subject switching not working**: User said "I want to talk about AP World History" but agent stayed on English
2. **Subject switching not working**: User said "I want to talk about Chemistry" but agent stayed on English  
3. **Interruption not working**: Agent not allowing user to interrupt and change topics

## Root Cause Analysis

### Issue 1 & 2: Too Sticky Subject Logic
The previous `route_subject_sticky()` function had overly aggressive stickiness rules:
- When user said "AP World History" → only 1 keyword match ("history")
- With 1-2 keyword matches → stuck to current subject
- Result: Stayed on English even with explicit subject mentions

### Issue 3: Missing Explicit Transition Detection  
The router only looked at keyword counts, not at explicit phrases like:
- "I want to talk about..."
- "Can we discuss..."
- "Let's switch to..."
- These phrases indicated clear user intent to change topics

## Solution Implemented

### Updated `route_subject_sticky()` Function
**File**: `src/router_agent.py`

Added detection for explicit transition phrases:
```python
has_explicit = any(
    re.search(pattern, text_lower)
    for pattern in [
        r"\b(want to|let'?s|can we|could we|should we)\s+(talk|discuss|go|switch|change)",
        r"\b(switch|change|go)\s+(to|back to)\b",
        r"\b(i'?d like to|i'?m interested in)\b",
        r"\b(now)\s+(let'?s|can we)\s+(talk|discuss)",
    ]
)
```

### New Routing Logic
1. **Score 0 keywords + current_subject**: Stick (e.g., "what about this?" while studying chemistry → stays chemistry)
2. **Explicit transition phrases detected**: ALWAYS allow switch (e.g., "I want to talk about X" → switches even with 1 keyword)
3. **Score 1 keyword + current subject + NO explicit phrase**: Stick (avoid accidental switches)
4. **Score 2+ keywords + different subject**: Switch (e.g., "Solve this quadratic equation while studying history" → switches to math)

## Integration Points

### Voice Agent (`src/voice_agent.py`)
- Already properly using `route_subject_sticky(user_input, self.current_subject)`
- Tracking `self.current_subject` to maintain context across turns
- Logging explicit switches with `[Subject Switch]` messages

### Barge-in Flow
- Interruptions now properly route using the fixed sticky logic
- When user interrupts with "I want to talk about X", agent:
  1. Detects interruption via `_listen_for_barge_in`
  2. Stops TTS with `self.mouth.stop(wait=False)`
  3. Routes new input with `route_subject_sticky(interruption, self.current_subject)`
  4. Explicit transition detected → switches subject
  5. Responds with new subject specialist

## Test Results

All tests passing for:
- ✅ Explicit transitions (want to, let's, can we, switch to)
- ✅ Subject switching from any subject to any other
- ✅ Sticky behavior preserved for weak follow-ups
- ✅ Barge-in interruption threshold detection
- ✅ Subject persistence across turns
- ✅ Integration with conversation memory per-subject

## Example Conversations Now Working

**Scenario 1: Direct Topic Switch**
```
User: "I want to talk about AP World History"
Agent: Switches to History specialist ✓
```

**Scenario 2: Multiple Switches**
```
User: "I want to talk about Chemistry" (from History)
Agent: Switches to Chemistry ✓

User: "Can you explain that more?"
Agent: Stays on Chemistry (sticky) ✓

User: "Let's switch to Math"
Agent: Switches to Math ✓
```

**Scenario 3: Interruption with Subject Change**
```
Agent: Speaking about Chemistry...
User: [Interrupts] "I want to talk about English"
Agent: Detects explicit "want to talk", stops speaking, switches to English ✓
```

## Files Modified
- `src/router_agent.py` - Enhanced `route_subject_sticky()` with explicit transition detection
- `src/voice_agent.py` - Already integrated, no changes needed (was waiting for router fix)

## Verification
Run these tests to verify everything works:
```bash
python test_subject_switch_fix.py      # Basic subject switching
python test_barge_in_flow.py           # Barge-in detection
python test_end_to_end_fix.py          # Full scenario simulation
```

All three tests should show ✅ results.
