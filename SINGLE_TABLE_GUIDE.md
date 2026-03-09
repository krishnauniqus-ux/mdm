# Data Quality - Single Table Format

## 📊 Interface

Everything in ONE table - like Excel!

```
┌────────────────────────────────────────────────────────────────────────┐
│  Metrics: Rows | Columns | Rejected | [Apply All] | [Undo]             │
├────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ✓ | Column | Sample Values | Mode | Pattern | Replace | Case | AI    │
│  ──┼────────┼───────────────┼──────┼─────────┼─────────┼──────┼────   │
│  ☑ | Name   | John, Jane... | Clean| [^a-z]  |         |      | 🤖    │
│  ──┼────────┼───────────────┼──────┼─────────┼─────────┼──────┼────   │
│  ☑ | Email  | test@...      | Valid| ^[a-z]  |         |      | 🤖    │
│  ──┼────────┼───────────────┼──────┼─────────┼─────────┼──────┼────   │
│  ☐ | Phone  | 123...        | Extr | [0-9]   |         |      | 🤖    │
│  ──┼────────┼───────────────┼──────┼─────────┼─────────┼──────┼────   │
│                                                                          │
├────────────────────────────────────────────────────────────────────────┤
│  [Enable All] [Disable All] [Preview] [Clear All]                      │
├────────────────────────────────────────────────────────────────────────┤
│  📜 History | 🗑️ Rejected                                              │
└────────────────────────────────────────────────────────────────────────┘
```

## 🚀 How to Use

1. **Check ✓** columns you want to transform
2. **Select Mode** from dropdown
3. **Enter Pattern** (for regex modes)
4. **Enter Replace** (for Replace mode)
5. **Select Case** (for Case mode)
6. **Click 🤖** for AI suggestion
7. **Click Apply All** at top

## 📋 Table Columns

| Column | Description |
|--------|-------------|
| ✓ | Enable/disable this transformation |
| Column | Column name |
| Sample Values | First 3 values from column |
| Mode | Clean/Replace/Extract/Validate/Case/Length |
| Pattern | Regex pattern or length (10 or 5-20) |
| Replace | Replacement text (Replace mode only) |
| Case | UPPER/lower/Title (Case mode only) |
| AI | Click for smart suggestion |

## 🎯 Modes

- **Clean**: Remove matching characters
- **Replace**: Find and replace
- **Extract**: Keep only matching
- **Validate**: Keep only rows that match
- **Case**: Change text case
- **Length**: Validate length

## 🤖 AI Button

Click 🤖 for any column to get smart suggestions:
- Detects underscores → Replace with space
- Detects special chars → Remove them
- Detects emails → Validate format
- Detects numbers → Extract digits

## ⚡ Quick Actions

- **Enable All**: Check all columns
- **Disable All**: Uncheck all columns
- **Preview**: See before/after samples
- **Clear All**: Reset all configurations
- **Apply All**: Apply all enabled transformations
- **Undo**: Revert last transformation

## 💡 Example

```
Row 1: ☑ | Name  | John@Doe | Clean | [^a-zA-Z0-9\s] | | | 🤖
Row 2: ☑ | Email | test@... | Valid | ^[a-z]+@...    | | | 🤖
Row 3: ☑ | Phone | 123-456  | Extr  | [0-9]          | | | 🤖

Click: Apply All
→ All 3 columns transformed!
```

## ✅ Features

- **Single table view**: Everything visible at once
- **Inline editing**: Configure directly in table
- **Multi-column**: Enable multiple, apply all
- **AI suggestions**: Smart pattern detection
- **Preview**: See changes before applying
- **Undo**: Revert last transformation
- **Simple**: No complex UI, just a table

---

**Simple, clean, and powerful!**
