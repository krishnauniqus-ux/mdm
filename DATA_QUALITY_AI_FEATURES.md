# Data Quality - New AI Features

## 🎉 What's New

### 1. Replace Mode (Find & Replace)
A new regex mode that allows you to find patterns and replace them with custom text.

**Use Cases**:
- Replace underscores with spaces: `hello_world` → `hello world`
- Replace special chars with hyphens: `hello@world` → `hello-world`
- Replace multiple spaces with single space
- Replace digits with placeholders

**Quick Actions**: One-click buttons for common replacements (Space, Empty, Hyphen, Underscore)

---

### 2. AI Regex Suggestions 🤖

Get intelligent regex pattern suggestions based on your data and requirements!

#### How to Use:
1. Select a column
2. Open "AI Regex Suggestions" expander
3. Describe what you want in plain English
4. Click "Get AI Suggestion"
5. Review the suggestion
6. Click "Use This Pattern" to apply

#### Example Prompts:
```
"Remove special characters and replace underscores with spaces"
"Extract only phone numbers"
"Validate email addresses"
"Replace multiple spaces with single space"
"Keep only alphanumeric characters"
```

#### What AI Provides:
- ✅ Recommended mode (Validate/Clean/Replace/Extract)
- ✅ Regex pattern
- ✅ Replacement text (if applicable)
- ✅ Clear explanation
- ✅ Confidence level
- ✅ Preview on sample data

#### Smart Detection:
The AI analyzes your column data and detects:
- Special characters
- Underscores
- Email patterns
- Phone numbers
- Spaces and formatting issues

---

## Complete Regex Modes

### 1. Validate (Keep matching)
- Keeps only rows where the ENTIRE value matches
- Invalid rows are rejected
- Example: `^\d{10}$` for 10-digit phone validation

### 2. Clean (Remove matching)
- Removes characters that match the pattern
- Example: `[^a-zA-Z0-9\s]` removes special characters

### 3. Replace (Find & Replace) ⭐ NEW
- Finds pattern and replaces with your text
- Example: `_` → ` ` replaces underscores with spaces

### 4. Extract (Keep only matching)
- Keeps only characters that match
- Example: `[0-9]` keeps only digits

---

## Common Workflows

### Workflow 1: Clean Names with Underscores
**Goal**: `hello_world@test` → `hello world test`

**Steps**:
1. Use AI: "Remove special characters and replace underscores with spaces"
2. AI suggests two-step process:
   - Step 1: Replace `_` with ` ` (space)
   - Step 2: Clean `[^a-zA-Z0-9\s]`
3. Apply both rules sequentially

### Workflow 2: Extract Phone Numbers
**Goal**: `Phone: 9876543210` → `9876543210`

**Steps**:
1. Use AI: "Extract only phone numbers"
2. AI suggests: Extract mode with pattern `[0-9]`
3. Apply rule

### Workflow 3: Standardize Formatting
**Goal**: `hello    world` → `hello world`

**Steps**:
1. Use AI: "Replace multiple spaces with single space"
2. AI suggests: Replace `\s+` with ` ` (space)
3. Apply rule

---

## Tips for Best Results

### For AI Suggestions:
- Be specific about what you want
- Mention both "remove" and "replace" if needed
- Use keywords like "validate", "extract", "clean", "replace"
- Review the suggestion before applying

### For Manual Patterns:
- Test with Preview first
- Check the Before/After columns
- Use the pattern examples as reference
- Start simple, then refine

### For Complex Transformations:
- Break into multiple steps
- Apply rules one at a time
- Use Undo if needed
- Check rejected records

---

## Pattern Library

### Cleaning Patterns
```regex
[^a-zA-Z0-9\s]     # Remove special chars
[^0-9]              # Remove non-digits
\s                  # Remove spaces
\s+                 # Remove multiple spaces
```

### Replacement Patterns
```regex
_        → (space)  # Underscore to space
\s+      → (space)  # Multiple spaces to one
[^a-zA-Z0-9\s] → -  # Special chars to hyphen
\d+      → [NUM]    # Digits to placeholder
```

### Validation Patterns
```regex
^\d{10}$                                              # 10-digit phone
^[A-Z]{5}[0-9]{4}[A-Z]{1}$                           # PAN card
^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$    # Email
```

### Extraction Patterns
```regex
[0-9]              # Extract digits
[a-zA-Z]           # Extract letters
[a-zA-Z0-9]        # Extract alphanumeric
```

---

## Troubleshooting

### AI Suggestion Not What You Expected?
- Refine your prompt with more details
- Try different keywords
- Check the sample data preview
- Manually adjust the pattern

### Pattern Not Working?
- Check for regex syntax errors
- Use Preview to test
- Verify the mode is correct
- Check pattern examples

### Need Multiple Transformations?
- Apply rules sequentially
- Use Undo if something goes wrong
- Check Applied Rules history
- Export rejected records to review

---

## Next Steps

1. Try the AI suggestions with your data
2. Experiment with different modes
3. Build a library of patterns for your use cases
4. Use the Undo feature to iterate safely
5. Export rejected records for review

For detailed pattern syntax, see `REGEX_PATTERN_GUIDE.md`
