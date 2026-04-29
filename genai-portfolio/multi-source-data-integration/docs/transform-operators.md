# Transform Operators Reference

The `TransformEngine` (`src/mapping/transform_engine.py`) implements 18 operators that can be composed in YAML mapping files to translate source columns into target columns. Each operator has a name, a parameter set, and well-defined input/output behavior.

## Operator Catalog

### `DIRECT`
**Description:** Copy the source value unchanged.

**Parameters:** none

**Example:**
```yaml
- target_column: CUSTOMER_NAME
  transform: DIRECT
  source_column: CUST_NM
```
**Input:** `"Acme Corp"` -> **Output:** `"Acme Corp"`

---

### `CONSTANT`
**Description:** Emit a fixed value regardless of input.

**Parameters:**
- `value` (required): the constant to emit

**Example:**
```yaml
- target_column: SOURCE_SYSTEM
  transform: CONSTANT
  params: { value: "ORACLE_ERP" }
```

---

### `CAST_TO_VARCHAR`
**Description:** Cast any value to its string representation; nulls become empty string.

**Parameters:** none

**Example:**
```yaml
- target_column: ACCOUNT_NUMBER_STR
  transform: CAST_TO_VARCHAR
  source_column: ACCT_NUM
```
**Input:** `12345` (int) -> **Output:** `"12345"`

---

### `CAST_TIMESTAMP`
**Description:** Parse strings or numeric epochs into ISO-8601 timestamps.

**Parameters:**
- `format` (optional): explicit `strptime` format. If omitted, common formats are tried.

**Example:**
```yaml
- target_column: CREATED_AT
  transform: CAST_TIMESTAMP
  source_column: CREATE_DT
  params: { format: "%Y-%m-%d %H:%M:%S" }
```
**Input:** `"2024-01-15 09:30:00"` -> **Output:** `"2024-01-15T09:30:00"`

---

### `CAST_DATE`
**Description:** Parse to a date (no time component).

**Parameters:**
- `format` (optional)

**Example:**
```yaml
- target_column: BIRTH_DATE
  transform: CAST_DATE
  source_column: DOB
```
**Input:** `"19850712"` -> **Output:** `"1985-07-12"`

---

### `CONCATENATE`
**Description:** Join multiple source columns with a separator.

**Parameters:**
- `source_columns` (required): list of column names
- `separator` (optional, default `" "`)

**Example:**
```yaml
- target_column: FULL_ADDRESS
  transform: CONCATENATE
  params:
    source_columns: [ADDR_LINE_1, ADDR_LINE_2, CITY, STATE, ZIP]
    separator: ", "
```
**Input:** `["123 Main St", "Apt 4", "Boston", "MA", "02101"]`
**Output:** `"123 Main St, Apt 4, Boston, MA, 02101"`

---

### `TRIM`
**Description:** Strip leading and trailing whitespace.

**Parameters:** none

**Example:**
```yaml
- target_column: EMAIL
  transform: TRIM
  source_column: EMAIL_ADDR
```
**Input:** `"  user@example.com  "` -> **Output:** `"user@example.com"`

---

### `TRIM_UPPER_FIRST`
**Description:** Trim whitespace, then uppercase the first character.

**Parameters:** none

**Example:**
```yaml
- target_column: STATUS_CODE
  transform: TRIM_UPPER_FIRST
  source_column: STATUS
```
**Input:** `"  active "` -> **Output:** `"Active"`

---

### `LOWERCASE`
**Description:** Convert string to lowercase.

**Parameters:** none

**Example:**
```yaml
- target_column: EMAIL_NORMALIZED
  transform: LOWERCASE
  source_column: EMAIL
```
**Input:** `"User@Example.COM"` -> **Output:** `"user@example.com"`

---

### `LOOKUP`
**Description:** Map source value through a lookup dictionary.

**Parameters:**
- `mapping` (required): dict of source_value -> target_value
- `default` (optional): value to use if source not in mapping

**Example:**
```yaml
- target_column: CUSTOMER_TYPE
  transform: LOOKUP
  source_column: CUST_TYPE_CD
  params:
    mapping: { "B": "Business", "I": "Individual", "G": "Government" }
    default: "Unknown"
```
**Input:** `"B"` -> **Output:** `"Business"`

---

### `NORMALIZE_PHONE`
**Description:** Convert phone numbers to E.164 format. Strips formatting, validates digits, prepends country code if missing.

**Parameters:**
- `default_country_code` (optional, default `"+1"`)

**Example:**
```yaml
- target_column: PHONE_E164
  transform: NORMALIZE_PHONE
  source_column: PHONE
  params: { default_country_code: "+1" }
```
**Inputs:**
- `"(617) 555-0123"` -> `"+16175550123"`
- `"617.555.0123"` -> `"+16175550123"`
- `"+44 20 7946 0958"` -> `"+442079460958"`

---

### `EXTRACT_FIRST_NAME`
**Description:** Extract first name from a full name string.

**Parameters:** none

**Example:**
```yaml
- target_column: FIRST_NAME
  transform: EXTRACT_FIRST_NAME
  source_column: FULL_NAME
```
**Inputs:**
- `"John A. Smith"` -> `"John"`
- `"Smith, John"` -> `"John"` (handles "Last, First" format)

---

### `EXTRACT_LAST_NAME`
**Description:** Extract last name from a full name string.

**Parameters:** none

**Example:**
```yaml
- target_column: LAST_NAME
  transform: EXTRACT_LAST_NAME
  source_column: FULL_NAME
```
**Inputs:**
- `"John A. Smith"` -> `"Smith"`
- `"Smith, John"` -> `"Smith"`

---

### `PAD_ISO_COUNTRY`
**Description:** Convert ISO 3166-1 alpha-2 codes to alpha-3.

**Parameters:** none

**Example:**
```yaml
- target_column: COUNTRY_CODE_ISO3
  transform: PAD_ISO_COUNTRY
  source_column: COUNTRY_CODE
```
**Inputs:**
- `"US"` -> `"USA"`
- `"GB"` -> `"GBR"`
- `"DE"` -> `"DEU"`

---

### `PREFIX`
**Description:** Prepend a string to the source value.

**Parameters:**
- `prefix` (required)

**Example:**
```yaml
- target_column: ACCOUNT_KEY
  transform: PREFIX
  source_column: ACCT_ID
  params: { prefix: "ACME-" }
```
**Input:** `"100245"` -> **Output:** `"ACME-100245"`

---

### `BIT_TO_BOOLEAN`
**Description:** Convert SQL Server BIT (0/1, "0"/"1", "Y"/"N") to Python boolean.

**Parameters:** none

**Example:**
```yaml
- target_column: IS_ACTIVE
  transform: BIT_TO_BOOLEAN
  source_column: ACTIVE_FLG
```
**Inputs:**
- `1` -> `True`
- `"0"` -> `False`
- `"Y"` -> `True`

---

### `CONDITIONAL`
**Description:** Apply different transforms or values based on a condition.

**Parameters:**
- `condition` (required): one of `equals`, `not_equals`, `is_null`, `is_not_null`
- `compare_value` (required for equals/not_equals)
- `if_true` (required): value or transform spec
- `if_false` (required): value or transform spec

**Example:**
```yaml
- target_column: DISPLAY_NAME
  transform: CONDITIONAL
  source_column: COMPANY_NAME
  params:
    condition: is_not_null
    if_true: { transform: DIRECT }
    if_false: { transform: CONSTANT, value: "Individual Customer" }
```

---

### `SURROGATE_KEY_LOOKUP`
**Description:** Look up the surrogate key for a natural key in a target dimension table. Caches lookups in memory for performance.

**Parameters:**
- `dimension_table` (required): target dim table name
- `natural_key_column` (required): natural key column in source
- `surrogate_key_column` (required): surrogate key column to return

**Example:**
```yaml
- target_column: CUSTOMER_SK
  transform: SURROGATE_KEY_LOOKUP
  source_column: CUSTOMER_NATURAL_KEY
  params:
    dimension_table: DIM_CUSTOMER
    natural_key_column: CUSTOMER_NATURAL_KEY
    surrogate_key_column: CUSTOMER_SK
```

---

## Composition

Multiple transforms can be chained for a single target column by listing them in execution order. The output of each transform feeds the next.

```yaml
- target_column: EMAIL_NORMALIZED
  source_column: EMAIL_ADDR
  transforms:
    - TRIM
    - LOWERCASE
```

## Adding a Custom Transform

To add a new operator:

1. Implement the function in `src/mapping/transform_engine.py`:
   ```python
   def _my_transform(self, value: Any, params: Optional[Dict] = None) -> Any:
       # implementation
       return transformed_value
   ```

2. Register it in `__init__`:
   ```python
   "MY_TRANSFORM": self._my_transform,
   ```

3. Add a unit test in `tests/test_schema_mapper.py` covering at least one happy path and one edge case.

4. Document it here.

## Testing

Each transform is covered by unit tests in `tests/test_schema_mapper.py::TestTransformEngine`. To add a regression test:

```python
def test_my_transform_handles_null(self) -> None:
    engine = TransformEngine()
    result = engine.apply("MY_TRANSFORM", None, {})
    assert result is None  # or expected behavior
```
