# Data Quality Rule Generator - Updates Summary

## Overview
Updated the DQ Rule Generator to follow specific mapping and formatting requirements for dimension classification and human-readable rule generation.

## Key Changes

### 1. Dimension Reclassification
**Requirement:** Any rule derived from technical metadata containing keywords such as CHAR, CHARACTER, VARCHAR, or VARCHAR2 must be assigned to "Character Length" dimension.

**Implementation:**
- Updated AI prompt in `generate_comprehensive_ai_prompt()` to explicitly map string length constraints to "Character Length" dimension
- Added `post_process_rules()` function to enforce dimension reclassification
- Keywords monitored: CHAR, CHARACTER, VARCHAR, VARCHAR2, STRING, TEXT, "max length", "maximum characters"

**Example:**
- Technical: `VARCHAR2(30)` → Dimension: `Character Length`
- Technical: `CHAR(19)` → Dimension: `Character Length`

### 2. Human-Readable Rule Formatting
**Requirement:** Transform technical constraints into natural language.

**Implementation:**
- Updated prompt to generate human-readable rule statements
- Format: `[Field Name] + should/must + condition`

**Examples:**
- Technical: `Field 'Account Name' is VARCHAR2(30)`
- Output: `Account Name should be maximum 30 characters`

### 3. Completeness Priority
**Requirement:** For every 'Mandatory' field, ensure a 'Completeness' dimension rule is generated.

**Implementation:**
- `post_process_rules()` checks if field is marked as mandatory (via `*` or keywords)
- Automatically generates Completeness rule if missing
- Format: `[Field Name] must not be blank`

**Example:**
- If field has `*` indicator or "mandatory" keyword
- Generated rule: `Account Name must not be blank` (Dimension: Completeness)

### 4. Consistency Check - Separate Line Items
**Requirement:** If a field has both 'Character Length' and 'Completeness' rules, generate them as separate line items.

**Implementation:**
- AI prompt instructs to generate separate rules for each dimension
- `post_process_rules()` ensures both rules exist when applicable
- Each rule is a separate row in the output DataFrame

**Example:**
For field `Account Name VARCHAR2(30) *`:
- Rule 1: Dimension = "Character Length", Rule = "Account Name should be maximum 30 characters"
- Rule 2: Dimension = "Completeness", Rule = "Account Name must not be blank"

## Modified Files

### 1. `features/rule_generator/engine.py`
- **Updated `generate_comprehensive_ai_prompt()`**: Enhanced prompt with explicit dimension mapping rules
- **Added `post_process_rules()`**: New function to enforce dimension reclassification and ensure completeness

### 2. `features/rule_generator/ui.py`
- **Updated `_generate_rules_with_comprehensive_engine()`**: Integrated `post_process_rules()` call
- **Updated `highlight_dimension()`**: Added bold styling for "Character Length" dimension

## Dimension Classification Rules

### Character Length
- Use for: String length constraints
- Keywords: CHAR, CHARACTER, VARCHAR, VARCHAR2, STRING, TEXT, "max length", "maximum characters"
- Format: `[Field] should be maximum X characters`

### Completeness
- Use for: Mandatory/required fields
- Keywords: `*`, mandatory, required, not null, must not be blank
- Format: `[Field] must not be blank`

### Validity
- Use for: Data type and format validation
- Format: `[Field] must be numeric`, `[Field] must be in [format]`

### Other Dimensions
- Uniqueness: Unique constraints
- Conformity: Format restrictions (NOT length)
- Accuracy: Value ranges and precision

## Testing Recommendations

1. **Test with VARCHAR/CHAR fields**: Verify dimension is "Character Length"
2. **Test with mandatory fields**: Verify Completeness rule is generated
3. **Test with mandatory + length constraint**: Verify two separate rules are generated
4. **Test human-readable format**: Verify technical constraints are transformed correctly

## Example Output

| S.No | Business Field | Dimension | Data Quality Rule |
|------|---------------|-----------|-------------------|
| 1 | Account Name | Character Length | Account Name should be maximum 30 characters |
| 2 | Account Name | Completeness | Account Name must not be blank |
| 3 | Invoice Date | Validity | Invoice Date must be in YYYY/MM/DD date format |
| 4 | Amount | Validity | Amount must be numeric |

## Notes

- The post-processing function runs after AI generation to ensure consistency
- All existing functionality is preserved
- The changes are backward compatible with existing rule generation logic
