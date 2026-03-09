# Data Quality - Regex Pattern Guide

## Understanding Regex Modes

The Data Quality tab supports four regex modes for different use cases:

### 1. Replace Mode (Find & Replace) ⭐ NEW
**Purpose**: Find characters matching the pattern and replace them with specified text

**How it works**: Characters matching the pattern are REPLACED with your replacement text

**Common Use Cases**:

#### Replace Underscores with Spaces
```regex
Pattern: _
Replace with: (space)
```
- Matches: Underscore character
- **Result**: Underscores become spaces
- **Example**: 
  - Before: `hello_world_test`
  - After: `hello world test`

#### Replace Special Characters with Hyphen
```regex
Pattern: [^a-zA-Z0-9\s]
Replace with: -
```
- Matches: Any special character
- **Result**: Special characters become hyphens
- **Example**:
  - Before: `hello@world#test`
  - After: `hello-world-test`

#### Replace Multiple Spaces with Single Space
```regex
Pattern: \s+
Replace with: (space)
```
- Matches: One or more consecutive spaces
- **Result**: Multiple spaces become single space
- **Example**:
  - Before: `hello    world`
  - After: `hello world`

#### Replace Digits with Placeholder
```regex
Pattern: \d+
Replace with: [NUM]
```
- Matches: One or more consecutive digits
- **Result**: Number sequences replaced with [NUM]
- **Example**:
  - Before: `Order123`
  - After: `Order[NUM]`

#### Remove Special Chars AND Replace Underscores
**Two-step process**:
1. First rule: Pattern `_`, Replace with ` ` (space)
2. Second rule: Pattern `[^a-zA-Z0-9\s]`, Mode: Clean
- **Example**:
  - Before: `hello_world@test#123`
  - After Step 1: `hello world@test#123`
  - After Step 2: `hello worldtest123`

---

### 2. Clean Mode (Remove matching)
### 2. Clean Mode (Remove matching)
**Purpose**: Remove characters that match the pattern

**How it works**: Characters matching the pattern are REMOVED from values

**Common Use Cases**:

#### Remove Special Characters
```regex
[^a-zA-Z0-9\s]
```
- **Explanation**: `[^...]` means "NOT these characters"
- Matches: Any character that is NOT a letter, digit, or space
- **Result**: Special characters are removed
- **Example**: 
  - Before: `Hello@World#123!`
  - After: `HelloWorld123`

#### Remove All Non-Digits
```regex
[^0-9]
```
- Matches: Any character that is NOT a digit
- **Result**: Only numbers remain
- **Example**:
  - Before: `Phone: 9876543210`
  - After: `9876543210`

#### Remove Spaces
```regex
\s
```
- Matches: Any whitespace character
- **Result**: All spaces removed
- **Example**:
  - Before: `Hello World`
  - After: `HelloWorld`

---

### 3. Extract Mode (Keep only matching)
**Purpose**: Keep only characters that match the pattern

**How it works**: Only characters matching the pattern are KEPT, everything else is removed

**Common Use Cases**:

#### Keep Only Digits
```regex
[0-9]
```
- Matches: Any digit
- **Result**: Only numbers are kept
- **Example**:
  - Before: `Price: $123.45`
  - After: `12345`

#### Keep Only Letters
```regex
[a-zA-Z]
```
- Matches: Any letter (uppercase or lowercase)
- **Result**: Only letters remain
- **Example**:
  - Before: `Hello123!`
  - After: `Hello`

#### Keep Alphanumeric
```regex
[a-zA-Z0-9]
```
- Matches: Any letter or digit
- **Result**: Only letters and numbers remain
- **Example**:
  - Before: `User@123!`
  - After: `User123`

---

### 4. Validate Mode (Full match)
**Purpose**: Keep only rows where the ENTIRE value matches the pattern

**How it works**: Rows that don't match are rejected (not transformed)

**Common Use Cases**:

#### Validate 10-Digit Phone
```regex
^\d{10}$
```
- `^` = start of string
- `\d{10}` = exactly 10 digits
- `$` = end of string
- **Result**: Only 10-digit values pass
- **Example**:
  - ✓ `9876543210` → PASS
  - ✗ `987654321` → REJECTED (only 9 digits)
  - ✗ `98765432100` → REJECTED (11 digits)

#### Validate Email
```regex
^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$
```
- **Result**: Only valid email formats pass
- **Example**:
  - ✓ `user@example.com` → PASS
  - ✗ `invalid.email` → REJECTED

#### Validate PAN Card
```regex
^[A-Z]{5}[0-9]{4}[A-Z]{1}$
```
- 5 uppercase letters + 4 digits + 1 uppercase letter
- **Result**: Only valid PAN format passes
- **Example**:
  - ✓ `ABCDE1234F` → PASS
  - ✗ `ABCD1234F` → REJECTED (only 4 letters at start)

---

## Quick Reference

| Goal | Mode | Pattern | Replacement | Example |
|------|------|---------|-------------|---------|
| Replace underscores with spaces | Replace | `_` | ` ` (space) | `hello_world` → `hello world` |
| Replace special chars with hyphen | Replace | `[^a-zA-Z0-9\s]` | `-` | `hello@world` → `hello-world` |
| Replace multiple spaces | Replace | `\s+` | ` ` (space) | `hello  world` → `hello world` |
| Remove special chars | Clean | `[^a-zA-Z0-9\s]` | - | `Hello@123!` → `Hello123` |
| Extract only numbers | Extract | `[0-9]` | - | `Price: $123` → `123` |
| Extract only letters | Extract | `[a-zA-Z]` | - | `Hello123` → `Hello` |
| Remove spaces | Clean | `\s` | - | `Hello World` → `HelloWorld` |
| Validate phone (10 digits) | Validate | `^\d{10}$` | - | Only 10-digit values pass |
| Validate email | Validate | `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$` | - | Only valid emails pass |

---

## 🤖 AI Regex Suggestions

The Data Quality tab now includes AI-powered regex suggestions! Here's how to use it:

### How It Works

1. **Select your column** in the Data Quality tab
2. **Click "AI Regex Suggestions"** expander
3. **Describe what you want** in plain English
4. **Click "Get AI Suggestion"** button
5. **Review the suggestion** with pattern, mode, and explanation
6. **Click "Use This Pattern"** to apply it automatically

### Example Prompts

- "Remove special characters and replace underscores with spaces"
- "Extract only phone numbers"
- "Validate email addresses"
- "Replace multiple spaces with single space"
- "Keep only alphanumeric characters"
- "Remove all non-digits"
- "Validate 10-digit phone numbers"
- "Replace special characters with hyphens"

### What AI Analyzes

The AI suggestion engine:
- Analyzes your column data (sample of 20 values)
- Detects patterns (special chars, underscores, emails, phones, etc.)
- Matches your requirement to common patterns
- Suggests the best mode and pattern
- Provides confidence level (high/medium/low)
- Shows preview of transformation on sample data

### Auto-Detection

If you don't provide a specific requirement, the AI will:
- Detect underscores and suggest replacement with spaces
- Detect special characters and suggest cleaning
- Detect phone numbers and suggest validation
- Provide generic cleaning suggestions

---

## Important Notes

1. **Clean vs Extract**: 
   - Clean: Pattern defines what to REMOVE
   - Extract: Pattern defines what to KEEP

2. **Character Classes**:
   - `[a-zA-Z]` = any letter
   - `[0-9]` or `\d` = any digit
   - `[^...]` = NOT these characters
   - `\s` = whitespace (space, tab, newline)

3. **Anchors** (for Validate mode):
   - `^` = start of string
   - `$` = end of string
   - Use both for exact matching

4. **Quantifiers**:
   - `{10}` = exactly 10 times
   - `{2,}` = 2 or more times
   - `+` = one or more times
   - `*` = zero or more times

---

## Testing Your Pattern

1. Select a column
2. Choose the appropriate mode
3. Enter your pattern
4. Click "Preview Validation"
5. Check the Before/After columns to verify the transformation
6. If correct, click "Apply Rule"
