# -*- coding: utf-8 -*-
import gradio as gr
import snowflake.connector
import pandas as pd
from datetime import datetime, timedelta
import re
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import time
import logging
import traceback # Import traceback for detailed error logging

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ========== SNOWFLAKE FUNCTIONS ==========
def get_snowflake_connection(user, password, account):
    """Establish connection to Snowflake"""
    try:
        conn = snowflake.connector.connect(
            user=user,
            password=password,
            account=account,
            authenticator='snowflake'
        )
        logging.info("Successfully connected to Snowflake.")
        return conn, "✅ Successfully connected!"
    except Exception as e:
        logging.error(f"Connection failed: {str(e)}")
        traceback.print_exc() # Print full traceback
        return None, f"❌ Connection failed: {str(e)}"\

def disconnect_snowflake(conn):
    """Close Snowflake connection"""
    if conn:
        conn.close()
        logging.info("Disconnected from Snowflake.")
    return None, "🔌 Disconnected successfully"

def get_databases(conn):
    """Get list of databases"""
    if not conn:
        logging.warning("No connection to get databases.")
        return []
    try:
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES")
        dbs = [row[1] for row in cursor.fetchall()]
        logging.info(f"Found databases: {dbs}")
        return dbs
    except Exception as e:
        logging.error(f"Error getting databases: {str(e)}")
        traceback.print_exc()
        return []

def get_schemas(conn, database):
    """Get schemas for specific database"""
    if not conn or not database:
        logging.warning("Missing connection or database to get schemas.")
        return []
    try:
        cursor = conn.cursor()
        cursor.execute(f"SHOW SCHEMAS IN DATABASE {database}")
        schemas = [row[1] for row in cursor.fetchall()]
        logging.info(f"Found schemas for {database}: {schemas}")
        return schemas
    except Exception as e:
        logging.error(f"Error getting schemas for {database}: {str(e)}")
        traceback.print_exc()
        return []

def get_tables(conn, database, schema):
    """Get tables for specific schema"""
    if not conn or not database or not schema:
        logging.warning("Missing connection, database, or schema to get tables.")
        return []
    try:
        cursor = conn.cursor()
        cursor.execute(f"SHOW TABLES IN SCHEMA {database}.{schema}")
        tables = [row[1] for row in cursor.fetchall()]
        # Filter out specific tables that are used for internal data (like ORDER_KPIS, TEST_CASES)
        # and add "All" for UI selection if needed elsewhere, but for DQ, we need specific tables.
        filtered_tables = [t for t in tables if t.upper() not in ('TEST_CASES', 'ORDER_KPIS')]
        logging.info(f"Found tables for {database}.{schema}: {filtered_tables}")
        return filtered_tables
    except Exception as e:
        logging.error(f"Error getting tables for {database}.{schema}: {str(e)}")
        traceback.print_exc()
        return []

def get_columns_for_table(conn, database, schema, table):
    """Utility function to get columns for a given table for UI dropdowns"""
    if not conn or not database or not schema or not table:
        logging.warning("Missing connection, database, schema, or table to get columns.")
        return []
    try:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT COLUMN_NAME
            FROM {database}.INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{schema}'
            AND TABLE_NAME = '{table}'
            ORDER BY ORDINAL_POSITION
        """)
        columns = [row[0] for row in cursor.fetchall()]
        logging.info(f"Found columns for {database}.{schema}.{table}: {columns}")
        return columns
    except Exception as e:
        logging.error(f"Error getting columns for {database}.{schema}.{table}: {str(e)}")
        traceback.print_exc()
        return []

def _get_column_details_for_dq(conn, database, schema, table):
    """
    Utility function to get column names and types for a given table,
    specifically for Data Quality checks.
    Returns a list of dictionaries: [{'name': 'col_name', 'type': 'col_type'}, ...]
    """
    if not conn or not database or not schema or not table:
        logging.warning("Missing connection, database, schema, or table to get column details for DQ.")
        return []
    try:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT COLUMN_NAME, DATA_TYPE
            FROM {database}.INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{schema}'
            AND TABLE_NAME = '{table}'
            ORDER BY ORDINAL_POSITION
        """)
        columns_details = [{'name': row[0], 'type': row[1].upper()} for row in cursor.fetchall()]
        logging.info(f"Found column details for {database}.{schema}.{table} (DQ): {columns_details}")
        return columns_details
    except Exception as e:
        logging.error(f"Error getting column details for {database}.{schema}.{table} (DQ): {str(e)}")
        traceback.print_exc()
        return []

def _categorize_columns_by_type(column_details_list: List[Dict]) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Categorizes columns into numeric, date, string, and all columns based on their data types.
    Returns: (all_column_names, numeric_column_names, date_column_names, string_column_names)
    """
    numeric_cols = []
    date_cols = []
    string_cols = []
    all_cols = []

    for col in column_details_list:
        col_name = col['name']
        col_type = col['type']
        all_cols.append(col_name)

        if "NUMBER" in col_type or "INT" in col_type or "FLOAT" in col_type or "DOUBLE" in col_type:
            numeric_cols.append(col_name)
        elif "DATE" in col_type or "TIMESTAMP" in col_type:
            date_cols.append(col_name)
        elif "VARCHAR" in col_type or "TEXT" in col_type or "STRING" in col_type:
            string_cols.append(col_name)
    return all_cols, numeric_cols, date_cols, string_cols


def clone_schema(conn, source_db, source_schema, target_schema):
    """Clone schema with improved error handling and reporting"""
    if not conn:
        return False, "❌ Not connected to Snowflake.", pd.DataFrame()
    if not source_db or not source_schema or not target_schema:
        return False, "⚠️ Please provide Source Database, Source Schema, and a Target Schema name.", pd.DataFrame()

    cursor = conn.cursor()
    try:
        # First check if source schema exists
        cursor.execute(f"SHOW SCHEMAS LIKE '{source_schema}' IN DATABASE {source_db}")
        if not cursor.fetchall():
            logging.error(f"Source schema {source_db}.{source_schema} doesn't exist.")
            return False, f"❌ Source schema {source_db}.{source_schema} doesn't exist", pd.DataFrame()

        # Execute clone command
        clone_sql = f"CREATE OR REPLACE SCHEMA {source_db}.{target_schema} CLONE {source_db}.{source_schema}"
        logging.info(f"Executing clone SQL: {clone_sql}")
        cursor.execute(clone_sql)

        # Verify clone was successful
        cursor.execute(f"SHOW SCHEMAS LIKE '{target_schema}' IN DATABASE {source_db}")
        if not cursor.fetchall():
            logging.error(f"Clone failed - target schema {source_db}.{target_schema} not created.")
            return False, f"❌ Clone failed - target schema not created", pd.DataFrame()

        # Get list of cloned tables
        cursor.execute(f"SHOW TABLES IN SCHEMA {source_db}.{source_schema}")
        source_tables = [row[1] for row in cursor.fetchall()]

        cursor.execute(f"SHOW TABLES IN SCHEMA {source_db}.{target_schema}")
        clone_tables = [row[1] for row in cursor.fetchall()]

        # Create summary DataFrame
        df_tables = pd.DataFrame({
            'Database': source_db,
            'Source Schema': source_schema,
            'Clone Schema': target_schema,
            'Source Tables': len(source_tables),
            'Cloned Tables': len(clone_tables),
            'Status': '✅ Success' if len(source_tables) == len(clone_tables) else '⚠️ Partial Success'
        }, index=[0])

        logging.info(f"Successfully mirrored schema {source_db}.{source_schema} to {source_db}.{target_schema}")
        return True, f"✅ Successfully Mirrored Schema {source_db}.{source_schema} to {source_db}.{target_schema}", df_tables
    except Exception as e:
        logging.error(f"Clone failed: {str(e)}")
        traceback.print_exc()
        return False, f"❌ Clone failed: {str(e)}", pd.DataFrame()

# New function for executing clone from UI
def execute_clone(conn, source_db, source_schema, target_schema):
    if not conn:
        return "❌ Not connected to Snowflake."
    if not source_db or not source_schema or not target_schema:
        return "⚠️ Please provide Source Database, Source Schema, and a Target Schema name."

    success, message, _ = clone_schema(conn, source_db, source_schema, target_schema)
    return message


def compare_table_differences(conn, db_name, source_schema, clone_schema):
    """Compare tables between schemas"""
    if not conn:
        return pd.DataFrame()
    cursor = conn.cursor()

    query = f"""
    WITH source_tables AS (
        SELECT table_name
        FROM {db_name}.information_schema.tables
        WHERE table_schema = '{source_schema}'
    ),
    clone_tables AS (
        SELECT table_name
        FROM {db_name}.information_schema.tables
        WHERE table_schema = '{clone_schema}'
    )
    SELECT
        COALESCE(s.table_name, c.table_name) AS table_name,
        CASE
            WHEN s.table_name IS NULL THEN 'Missing in source - Table Dropped'
            WHEN c.table_name IS NULL THEN 'Missing in clone - Table Added'
            ELSE 'Present in both'
        END AS difference
    FROM source_tables s
    FULL OUTER JOIN clone_tables c ON s.table_name = c.table_name
    WHERE s.table_name IS NULL OR c.table_name IS NULL
    ORDER BY difference, table_name;
    """
    try:
        cursor.execute(query)
        results = cursor.fetchall()
        df = pd.DataFrame(results, columns=['Table', 'Difference'])
        logging.info(f"Table differences found: {len(df)} rows.")
        return df
    except Exception as e:
        logging.error(f"Error comparing table differences: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()

def compare_column_differences(conn, db_name, source_schema, clone_schema):
    """Compare columns and data types between schemas"""
    if not conn:
        return pd.DataFrame(), pd.DataFrame()
    cursor = conn.cursor()

    # Get common tables
    common_tables_query = f"""
    SELECT s.table_name
    FROM {db_name}.information_schema.tables s
    JOIN {db_name}.information_schema.tables c
        ON s.table_name = c.table_name
    WHERE s.table_schema = '{source_schema}'
    AND c.table_schema = '{clone_schema}';
    """
    try:
        cursor.execute(common_tables_query)
        common_tables = [row[0] for row in cursor.fetchall()]
    except Exception as e:
        logging.error(f"Error getting common tables for column comparison: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()

    column_diff_data = []
    datatype_diff_data = []

    for table in common_tables:
        try:
            # Get source table description
            cursor.execute(f"DESCRIBE TABLE {db_name}.{source_schema}.{table}")
            source_desc = cursor.fetchall()
            source_cols = {row[0]: row[1] for row in source_desc}

            # Get clone table description
            cursor.execute(f"DESCRIBE TABLE {db_name}.{clone_schema}.{table}")
            clone_desc = cursor.fetchall()
            clone_cols = {row[0]: row[1] for row in clone_desc}

            # Get all unique column names
            all_columns = set(source_cols.keys()).union(set(clone_cols.keys()))

            for col in all_columns:
                source_exists = col in source_cols
                clone_exists = col in clone_cols

                if not source_exists:
                    column_diff_data.append({
                        'Table': table,
                        'Column': col,
                        'Difference': 'Missing in source - Column Dropped',
                        'Source Data Type': None,
                        'Clone Data Type': clone_cols.get(col)
                    })
                elif not clone_exists:
                    column_diff_data.append({
                        'Table': table,
                        'Column': col,
                        'Difference': 'Missing in clone - Column Added',
                        'Source Data Type': source_cols.get(col),
                        'Clone Data Type': None
                    })
                else:
                    # Column exists in both - check data type
                    if source_cols[col] != clone_cols[col]:
                        datatype_diff_data.append({
                            'Table': table,
                            'Column': col,
                            'Source Data Type': source_cols[col],
                            'Clone Data Type': clone_cols[col],
                            'Difference': 'Data Type Changed'
                        })
        except Exception as e:
            logging.error(f"Error processing table {table} for column differences: {str(e)}")
            traceback.print_exc()
            continue # Continue to next table if one fails

    # Create DataFrames
    column_diff_df = pd.DataFrame(column_diff_data)
    if not column_diff_df.empty:
        column_diff_df = column_diff_df[['Table', 'Column', 'Difference', 'Source Data Type', 'Clone Data Type']]

    datatype_diff_df = pd.DataFrame(datatype_diff_data)
    if not datatype_diff_df.empty:
        datatype_diff_df = datatype_diff_df[['Table', 'Column', 'Source Data Type', 'Clone Data Type', 'Difference']]

    logging.info(f"Column differences found: {len(column_diff_df)} rows. Datatype differences found: {len(datatype_diff_df)} rows.")
    return column_diff_df, datatype_diff_df

def validate_kpis(conn, database, source_schema, target_schema):
    """Validate KPIs between source and clone schemas"""
    if not conn:
        return pd.DataFrame(), "❌ Not connected to Snowflake."
    cursor = conn.cursor()
    results = []

    try:
        # Fetch all KPI definitions
        kpi_query = f"SELECT KPI_ID, KPI_NAME, KPI_VALUE FROM {database}.{source_schema}.ORDER_KPIS"
        logging.info(f"Fetching KPIs with query: {kpi_query}")
        cursor.execute(kpi_query)
        kpis = cursor.fetchall()

        if not kpis:
            logging.warning("No KPIs found in ORDER_KPIS table.")
            return pd.DataFrame(), "⚠️ No KPIs found in ORDER_KPIS table."

        # First verify both schemas have the ORDER_DATA table
        source_has_table = False
        target_has_table = False
        try:
            cursor.execute(f"SELECT 1 FROM {database}.{source_schema}.ORDER_DATA LIMIT 1")
            source_has_table = True
        except Exception as e:
            logging.warning(f"ORDER_DATA table not found in source schema {source_schema}: {e}")
            traceback.print_exc()

        try:
            cursor.execute(f"SELECT 1 FROM {database}.{target_schema}.ORDER_DATA LIMIT 1")
            target_has_table = True
        except Exception as e:
            logging.warning(f"ORDER_DATA table not found in target schema {target_schema}: {e}")
            traceback.print_exc()

        if not source_has_table or not target_has_table:
            error_msg = "ORDER_DATA table missing in "
            if not source_has_table and not target_has_table:
                error_msg += "both schemas"
            elif not source_has_table:
                error_msg += "source schema"
            else:
                error_msg += "target schema"

            for kpi_id, kpi_name, kpi_sql in kpis:
                results.append({
                    'KPI ID': kpi_id,
                    'KPI Name': kpi_name,
                    'Source Value': f"ERROR: {error_msg}",
                    'Clone Value': f"ERROR: {error_msg}",
                    'Difference': "N/A",
                    'Diff %': "N/A",
                    'Status': "❌ Error"
                })
            logging.error(f"KPI validation failed - missing ORDER_DATA table: {error_msg}")
            return pd.DataFrame(results), "❌ Validation failed - missing ORDER_DATA table", gr.Button(visible=False)

        for kpi_id, kpi_name, kpi_sql in kpis:
            result_source = "N/A"
            result_clone = "N/A"

            try:
                # For source schema - replace only unqualified table names
                source_query = re.sub(r'\bORDER_DATA\b', f'{database}.{source_schema}.ORDER_DATA', kpi_sql, flags=re.IGNORECASE)
                logging.info(f"Executing source KPI query for {kpi_name}: {source_query}")
                cursor.execute(source_query)
                result_source = cursor.fetchone()[0] if cursor.rowcount > 0 else None
            except Exception as e:
                result_source = f"QUERY_ERROR: {str(e)}"
                logging.error(f"Error executing source KPI query for {kpi_name}: {e}")
                traceback.print_exc()

            try:
                # For target schema - replace only unqualified table names
                clone_query = re.sub(r'\bORDER_DATA\b', f'{database}.{target_schema}.ORDER_DATA', kpi_sql, flags=re.IGNORECASE)
                logging.info(f"Executing clone KPI query for {kpi_name}: {clone_query}")
                cursor.execute(clone_query)
                result_clone = cursor.fetchone()[0] if cursor.rowcount > 0 else None
            except Exception as e:
                result_clone = f"QUERY_ERROR: {str(e)}"
                logging.error(f"Error executing clone KPI query for {kpi_name}: {e}")
                traceback.print_exc()

            # Calculate differences if possible
            diff = "N/A"
            pct_diff = "N/A"
            status = "⚠️ Mismatch"

            try:
                if (isinstance(result_source, (int, float)) and isinstance(result_clone, (int, float))):
                    diff = float(result_source) - float(result_clone)
                    if float(result_source) != 0:
                        pct_diff = (diff / float(result_source)) * 100
                    else:
                        pct_diff = "N/A"
                    status = '✅ Match' if diff == 0 else '⚠️ Mismatch'
                elif str(result_source) == str(result_clone):
                    status = '✅ Match'
            except Exception as e:
                logging.warning(f"Error comparing KPI values for {kpi_name}: {e}. Setting diff/pct_diff to N/A.")
                traceback.print_exc()
                pass # Keep diff/pct_diff as N/A and status as Mismatch

            results.append({
                'KPI ID': kpi_id,
                'KPI Name': kpi_name,
                'Source Value': result_source,
                'Clone Value': result_clone,
                'Difference': diff if not isinstance(diff, float) else round(diff, 2),
                'Diff %': f"{round(pct_diff, 2)}%" if isinstance(pct_diff, float) else pct_diff,
                'Status': status
            })

        df = pd.DataFrame(results)
        logging.info("KPI validation completed.")
        return df, "✅ KPI validation completed", gr.Button(visible=True)

    except Exception as e:
        logging.error(f"KPI validation failed: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame(), f"❌ KPI validation failed: {str(e)}", gr.Button(visible=False)

# ===== TEST CASE VALIDATION FUNCTIONS =====
def verify_table_access(conn, database, schema, table_name):
    """Verifies if the current connection has access to the specified table."""
    cursor = conn.cursor()
    try:
        logging.info(f"Verifying access to {database}.{schema}.{table_name}")
        cursor.execute(f'SELECT 1 FROM {database}.{schema}.{table_name} LIMIT 1')
        return True
    except Exception as e:
        logging.warning(f"Access verification failed for {database}.{schema}.{table_name}: {str(e)}")
        traceback.print_exc()
        return False

def get_test_case_tables(conn, database, schema):
    """Get distinct tables from test cases table with error handling"""
    if not conn or not database or not schema:
        return ["All"]
    try:
        cursor = conn.cursor()
        # First verify TEST_CASES table exists
        cursor.execute(f"""
            SELECT COUNT(*)
            FROM {database}.information_schema.tables
            WHERE table_schema = '{schema}'
            AND table_name = 'TEST_CASES'
        """)
        if cursor.fetchone()[0] == 0:
            logging.warning(f"TEST_CASES table not found in {database}.{schema}")
            return ["All"]

        # Now get distinct tables
        cursor.execute(f"""
            SELECT DISTINCT TABLE_NAME
            FROM {database}.{schema}.TEST_CASES
            WHERE TABLE_NAME IS NOT NULL
            ORDER BY TABLE_NAME
        """)
        tables = [row[0] for row in cursor.fetchall()]
        logging.info(f"Found test case tables: {tables}")
        return ["All"] + tables
    except Exception as e:
        logging.error(f"Error getting test case tables: {str(e)}")
        traceback.print_exc()
        return ["All"]

def get_test_cases(conn, database, schema, table):
    """Get test cases for specific table with error handling"""
    if not conn or not database or not schema:
        return []
    try:
        cursor = conn.cursor()

        # First verify the TEST_CASES table exists
        cursor.execute(f"""
            SELECT COUNT(*)
            FROM {database}.information_schema.tables
            WHERE table_schema = '{schema}'
            AND table_name = 'TEST_CASES'
        """)
        if cursor.fetchone()[0] == 0:
            logging.warning(f"TEST_CASES table not found in {database}.{schema}")
            return []

        # Now fetch test cases
        if table == "All":
            query = f"""
                SELECT
                    TEST_CASE_ID,
                    TEST_ABBREVIATION,
                    TABLE_NAME,
                    TEST_DESCRIPTION,
                    SQL_CODE,
                    EXPECTED_RESULT
                FROM {database}.{schema}.TEST_CASES
                ORDER BY TEST_CASE_ID
            """
        else:
            query = f"""
                SELECT
                    TEST_CASE_ID,
                    TEST_ABBREVIATION,
                    TABLE_NAME,
                    TEST_DESCRIPTION,
                    SQL_CODE,
                    EXPECTED_RESULT
                FROM {database}.{schema}.TEST_CASES
                WHERE TABLE_NAME = '{table}'
                ORDER BY TEST_CASE_ID
            """
        logging.info(f"Fetching test cases with query: {query}")
        cursor.execute(query)
        cases = cursor.fetchall()
        logging.info(f"Found {len(cases)} test cases for {database}.{schema}.{table}")
        return cases
    except Exception as e: # Added this except block
        logging.error(f"Error getting test cases: {str(e)}")
        traceback.print_exc()
        return []

def validate_test_cases(conn, database, schema, test_cases):
    """Executes selected test cases and returns results."""
    if not conn:
        return pd.DataFrame(), "❌ Not connected to Snowflake.", gr.Button(visible=False)
    if not test_cases:
        return pd.DataFrame(), "⚠️ No test cases selected", gr.Button(visible=False)

    cursor = conn.cursor()
    results = []

    for case in test_cases:
        test_id, abbrev, table_name, desc, sql, expected = case
        expected = str(expected).strip()
        logging.info(f"Validating test case: {abbrev} for table {table_name}")

        # Verify table access first
        if not verify_table_access(conn, database, schema, table_name):
            results.append({
                'TEST CASE': abbrev,
                'CATEGORY': table_name,
                'EXPECTED RESULT': expected,
                'ACTUAL RESULT': f"ACCESS DENIED: No permissions on {database}.{schema}.{table_name}",
                'STATUS': "❌ PERMISSION ERROR"
            })
            continue

        try:
            # Modify SQL to use fully qualified names for the target table
            # This regex will replace 'table_name' only if it's a whole word, ignoring case.
            # It assumes the SQL_CODE in TEST_CASES uses the unqualified table name.
            qualified_sql = re.sub(
                rf'\b{re.escape(table_name)}\b',
                f'{database}.{schema}.{table_name}',
                sql,
                flags=re.IGNORECASE
            )
            logging.info(f"Executing test case SQL: {qualified_sql}")
            cursor.execute(qualified_sql)
            result = cursor.fetchone()
            actual_result = str(result[0]) if result else "0"

            status = "✅ PASS" if actual_result == expected else "❌ FAIL"
            results.append({
                'TEST CASE': abbrev,
                'CATEGORY': table_name,
                'EXPECTED RESULT': expected,
                'ACTUAL RESULT': actual_result,
                'STATUS': status
            })
            logging.info(f"Test case {abbrev} status: {status}")

        except Exception as e:
            error_msg = str(e).split('\n')[0] # Get only the first line of the error
            results.append({
                'TEST CASE': abbrev,
                'CATEGORY': table_name,
                'EXPECTED RESULT': expected,
                'ACTUAL RESULT': f"QUERY ERROR: {error_msg}",
                'STATUS': "❌ EXECUTION ERROR"
            })
            logging.error(f"Error executing test case {abbrev}: {e}")
            traceback.print_exc()

    df = pd.DataFrame(results)
    return df, "✅ Validation completed", gr.Button(visible=True)

# ===== DATA QUALITY VALIDATION FUNCTIONS =====
class DataQualityValidator:
    def __init__(self, conn):
        self.conn = conn

    def _execute_query(self, query):
        cursor = self.conn.cursor()
        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    def _get_table_columns(self, database, schema, table):
        """Returns list of column names and their types as fetched by _get_column_details_for_dq."""
        return _get_column_details_for_dq(self.conn, database, schema, table)

    def _get_column_details(self, database, schema, table):
        """
        Returns a dictionary mapping column name to its details (including type).
        Example: {'COLUMN_A': {'name': 'COLUMN_A', 'type': 'VARCHAR'}, ...}
        """
        query = f"SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE FROM {database}.INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table}'"
        df = self._execute_query(query)
        return {row['COLUMN_NAME'].upper(): row.to_dict() for _, row in df.iterrows()}

    def _run_row_count_check(self, database, schema, table, min_rows):
        query = f"SELECT COUNT(*) FROM {database}.{schema}.{table}"
        count = self._execute_query(query).iloc[0, 0]
        status = "✅ Pass" if count >= min_rows else "❌ Fail"
        details = f"Actual rows: {count}, Minimum expected: {min_rows}"
        return {"Check": "Row Count Check", "Column Name": "N/A", "Expected": f">= {min_rows}", "Actual": count, "Status": status, "Details": details}

    def _run_duplicate_rows_check(self, database, schema, table):
        columns_details = self._get_table_columns(database, schema, table)
        if not columns_details:
            return {"Check": "Duplicate Rows Check", "Column Name": "All Columns", "Expected": "0", "Actual": "N/A", "Status": "⚠️ N/A", "Details": "No columns found to check for duplicates."}

        # Check for duplicates across all columns
        cols_str = ", ".join([f'"{col["name"]}"' for col in columns_details]) # Quote column names
        query = f"""
        SELECT COUNT(*)
        FROM (
            SELECT {cols_str}
            FROM {database}.{schema}.{table}
            GROUP BY {cols_str}
            HAVING COUNT(*) > 1
        )
        """
        duplicate_count = self._execute_query(query).iloc[0, 0]
        status = "✅ Pass" if duplicate_count == 0 else "❌ Fail"
        details = f"Number of duplicate rows: {duplicate_count}"
        return {"Check": "Duplicate Rows Check", "Column Name": "All Columns", "Expected": "0", "Actual": duplicate_count, "Status": status, "Details": details}

    def _run_column_null_percentage_check(self, database, schema, table, selected_columns, threshold):
        results = []
        all_cols_details = _get_column_details_for_dq(self.conn, database, schema, table)
        all_column_names = [col['name'] for col in all_cols_details]

        # Logic: If selected_columns is empty, check all columns. Otherwise, check only selected columns.
        columns_to_check = selected_columns if selected_columns else all_column_names

        total_rows_query = f"SELECT COUNT(*) FROM {database}.{schema}.{table}"
        total_rows = self._execute_query(total_rows_query).iloc[0, 0]

        if total_rows == 0:
            results.append({"Check": "Column Null % Check", "Column Name": "N/A", "Expected": "N/A", "Actual": "N/A", "Status": "⚠️ N/A", "Details": "Table is empty, cannot check null percentage."})
            return results

        for col in columns_to_check:
            # Ensure column exists in table
            if col not in all_column_names:
                results.append({"Check": f"Column Null %", "Column Name": col, "Expected": f"<= {threshold}%", "Actual": "N/A", "Status": "❌ Error", "Details": f"Column '{col}' not found in table."})
                continue

            query = f"SELECT COUNT(*) FROM {database}.{schema}.{table} WHERE {col} IS NULL"
            null_count = self._execute_query(query).iloc[0, 0]

            null_percentage = (null_count / total_rows) * 100 if total_rows > 0 else 0
            status = "✅ Pass" if null_percentage <= threshold else "❌ Fail"
            details = f"Null count: {null_count}, Total rows: {total_rows}, Threshold: {threshold}%"
            results.append({"Check": f"Column Null %", "Column Name": col, "Expected": f"<= {threshold}%", "Actual": f"{null_percentage:.2f}%", "Status": status, "Details": details})
        return results

    def _run_table_overall_null_percentage_check(self, database, schema, table, threshold):
        all_cols_details = _get_column_details_for_dq(self.conn, database, schema, table)
        all_columns = [col['name'] for col in all_cols_details]

        if not all_columns:
            return {"Check": "Table Overall Null % Check", "Column Name": "All Columns", "Expected": f"<= {threshold}%", "Actual": "N/A", "Status": "⚠️ N/A", "Details": "No columns found in table."}

        total_rows_query = f"SELECT COUNT(*) FROM {database}.{schema}.{table}"
        total_rows = self._execute_query(total_rows_query).iloc[0, 0]

        if total_rows == 0:
            return {"Check": "Table Overall Null % Check", "Column Name": "All Columns", "Expected": f"<= {threshold}%", "Actual": "N/A", "Status": "⚠️ N/A", "Details": "Table is empty, cannot check overall null percentage."}

        total_null_cells = 0
        total_cells = total_rows * len(all_columns)

        for col in all_columns:
            query = f"SELECT COUNT(*) FROM {database}.{schema}.{table} WHERE {col} IS NULL"
            null_count = self._execute_query(query).iloc[0, 0]
            total_null_cells += null_count

        overall_null_percentage = (total_null_cells / total_cells) * 100 if total_cells > 0 else 0
        status = "✅ Pass" if overall_null_percentage <= threshold else "❌ Fail"
        details = f"Total null cells: {total_null_cells}, Total cells: {total_cells}, Threshold: {threshold}%"
        return {"Check": "Table Overall Null % Check", "Column Name": "All Columns", "Expected": f"<= {threshold}%", "Actual": f"{overall_null_percentage:.2f}%", "Status": status, "Details": details}

    def _run_value_range_check(self, database, schema, table, value_range_config):
        results = []
        # Skip if no config or all rows are empty/invalid
        if not value_range_config or all(not str(row[0]).strip() and not str(row[1]).strip() and not str(row[2]).strip() for row in value_range_config):
            return [] # Return empty list to indicate skipping this check

        all_cols_details_map = self._get_column_details(database, schema, table) # Get detailed column info once

        for row in value_range_config:
            col_name = str(row[0]).strip()
            min_val_str = str(row[1]).strip()
            max_val_str = str(row[2]).strip()

            if not col_name:
                continue # Skip this specific row if column name is empty

            # Skip this row if both min and max values are not provided
            if not min_val_str and not max_val_str:
                continue

            try:
                col_detail = all_cols_details_map.get(col_name.upper())
                if not col_detail:
                    results.append({"Check": f"Value Range", "Column Name": col_name, "Expected": "N/A", "Actual": "N/A", "Status": "❌ Error", "Details": f"Column '{col_name}' not found."})
                    continue
                col_type = col_detail['DATA_TYPE'].upper()

                if ("NUMBER" not in col_type and "INT" not in col_type and "FLOAT" not in col_type and "DOUBLE" not in col_type):
                     results.append({"Check": f"Value Range", "Column Name": col_name, "Expected": "Numeric Type", "Actual": col_type, "Status": "❌ Error", "Details": f"Column '{col_name}' is not a numeric type ({col_type})."})
                     continue

                min_val = float(min_val_str) if min_val_str else None
                max_val = float(max_val_str) if max_val_str else None

                where_clauses = []
                if min_val is not None:
                    where_clauses.append(f"{col_name} < {min_val}")
                if max_val is not None:
                    where_clauses.append(f"{col_name} > {max_val}")

                violation_query = f"SELECT COUNT(*) FROM {database}.{schema}.{table} WHERE {' OR '.join(where_clauses)}"
                violation_count = self._execute_query(violation_query).iloc[0, 0]

                status = "✅ Pass" if violation_count == 0 else "❌ Fail"
                expected_range_str = f"[{min_val_str if min_val_str else '-inf'}, {max_val_str if max_val_str else '+inf'}]"
                details = f"Violations: {violation_count}. Expected range: {expected_range_str}"
                results.append({"Check": f"Value Range", "Column Name": col_name, "Expected": expected_range_str, "Actual": violation_count, "Status": status, "Details": details})
            except ValueError:
                results.append({"Check": f"Value Range", "Column Name": col_name, "Expected": "Valid Numbers", "Actual": f"Min: '{min_val_str}', Max: '{max_val_str}'", "Status": "❌ Error", "Details": f"Invalid numeric range for column '{col_name}'."})
            except Exception as e:
                results.append({"Check": f"Value Range", "Column Name": col_name, "Expected": "Successful Query", "Actual": "Query Error", "Status": "❌ Error", "Details": f"Query error for '{col_name}': {str(e).splitlines()[0]}"})
        return results

    def _run_date_range_check(self, database, schema, table, date_range_config):
        results = []
        # Skip if no config or all rows are empty/invalid
        if not date_range_config or all(not str(row[0]).strip() and not str(row[1]).strip() and not str(row[2]).strip() for row in date_range_config):
            return [] # Return empty list to indicate skipping this check

        all_cols_details_map = self._get_column_details(database, schema, table) # Get detailed column info once

        for row in date_range_config:
            col_name = str(row[0]).strip()
            min_date_str = str(row[1]).strip()
            max_date_str = str(row[2]).strip()

            if not col_name:
                continue # Skip this specific row if column name is empty

            # Skip this row if both min and max dates are not provided
            if not min_date_str and not max_date_str:
                continue

            try:
                col_detail = all_cols_details_map.get(col_name.upper())
                if not col_detail:
                    results.append({"Check": f"Date Range", "Column Name": col_name, "Expected": "N/A", "Actual": "N/A", "Status": "❌ Error", "Details": f"Column '{col_name}' not found."})
                    continue
                col_type = col_detail['DATA_TYPE'].upper()

                if "DATE" not in col_type and "TIMESTAMP" not in col_type:
                    results.append({"Check": f"Date Range", "Column Name": col_name, "Expected": "Date/Timestamp Type", "Actual": col_type, "Status": "❌ Error", "Details": f"Column '{col_name}' is not a date/timestamp type ({col_type})."})
                    continue

                min_date = f"'{min_date_str}'" if min_date_str else None
                max_date = f"'{max_date_str}'" if max_date_str else None

                where_clauses = []
                if min_date is not None:
                    where_clauses.append(f"{col_name} < {min_date}")
                if max_date is not None:
                    where_clauses.append(f"{col_name} > {max_date}")

                violation_query = f"SELECT COUNT(*) FROM {database}.{schema}.{table} WHERE {' OR '.join(where_clauses)}"
                violation_count = self._execute_query(violation_query).iloc[0, 0]

                status = "✅ Pass" if violation_count == 0 else "❌ Fail"
                expected_range_str = f"[{min_date_str if min_date_str else '-inf'}, {max_date_str if max_date_str else '+inf'}]"
                details = f"Violations: {violation_count}. Expected range: {expected_range_str}"
                results.append({"Check": f"Date Range", "Column Name": col_name, "Expected": expected_range_str, "Actual": violation_count, "Status": status, "Details": details})
            except Exception as e:
                results.append({"Check": f"Date Range", "Column Name": col_name, "Expected": "Valid Dates", "Actual": f"Min: '{min_date_str}', Max: '{max_date_str}'", "Status": "❌ Error", "Details": f"Query error for '{col_name}': {str(e).splitlines()[0]}"})
        return results

    def _run_regex_pattern_check(self, database, schema, table, selected_columns, pattern):
        results = []
        all_cols_details = _get_column_details_for_dq(self.conn, database, schema, table)
        all_string_cols = [col['name'] for col in all_cols_details if "VARCHAR" in col['type'] or "TEXT" in col['type'] or "STRING" in col['type']]

        if not pattern:
            # Skip if no pattern is provided
            return []

        if not selected_columns:
            # If checkbox is selected but no specific columns are chosen, skip this check
            return []

        columns_to_check = selected_columns # Only check explicitly selected columns

        if not columns_to_check: # This might happen if selected_columns was empty after filtering, though the above check should catch it.
            return []

        for col in columns_to_check:
            # Ensure column exists and is a string type
            col_meta = next((item for item in all_cols_details if item["name"] == col), None)
            if not col_meta or ("VARCHAR" not in col_meta['type'] and "TEXT" not in col_meta['type'] and "STRING" not in col_meta['type']):
                results.append({"Check": f"Regex Pattern", "Column Name": col, "Expected": f"String Type", "Actual": f"Type: {col_meta['type'] if col_meta else 'N/A'}", "Status": "❌ Error", "Details": f"Column '{col}' not found or not a string type."})
                continue

            # Snowflake's RLIKE is case-sensitive by default, use ILIKE for case-insensitivity if desired
            # For this example, we'll assume case-sensitive RLIKE based on typical regex use
            query = f"SELECT COUNT(*) FROM {database}.{schema}.{table} WHERE {col} NOT RLIKE '{pattern}'"
            violation_count = self._execute_query(query).iloc[0, 0]
            status = "✅ Pass" if violation_count == 0 else "❌ Fail"
            details = f"Violations: {violation_count}. Pattern: {pattern}"
            results.append({"Check": f"Regex Pattern", "Column Name": col, "Expected": f"Match pattern '{pattern}'", "Actual": f"{violation_count} violations", "Status": status, "Details": details})
        return results

    def _run_foreign_key_check(self, database, schema, table, fk_column, ref_table, ref_column):
        if not (fk_column and ref_table and ref_column):
            return None # Return None to indicate skipping this check

        query = f"""
        SELECT COUNT(*)
        FROM {database}.{schema}.{table} AS t1
        LEFT JOIN {database}.{schema}.{ref_table} AS t2
            ON t1.{fk_column} = t2.{ref_column}
        WHERE t2.{ref_column} IS NULL AND t1.{fk_column} IS NOT NULL
        """
        violation_count = self._execute_query(query).iloc[0, 0]
        status = "✅ Pass" if violation_count == 0 else "❌ Fail"
        details = f"Unmatched FKs: {violation_count}. FK: {table}.{fk_column} -> {ref_table}.{ref_column}"
        return {"Check": "Foreign Key Check", "Column Name": fk_column, "Expected": f"FK to {ref_table}.{ref_column} exists", "Actual": f"{violation_count} unmatched FKs", "Status": status, "Details": details}

    def run_dq_checks(self, conn, database, schema, table,
                      dq_selected_columns_null, dq_null_threshold,
                      dq_check_table_overall_null_pct, dq_table_null_threshold,
                      dq_value_range_table, dq_date_range_table, dq_selected_columns_regex,
                      dq_check_row_count, dq_min_rows, dq_check_duplicate_rows,
                      dq_check_value_range, dq_check_date_range, dq_check_regex_pattern, dq_pattern,
                      dq_check_fk, dq_fk_column, dq_fk_ref_table, dq_fk_ref_column,
                      dq_check_column_null_pct):

        self.conn = conn # Ensure the connection is updated

        detailed_results = []
        status_message = "✅ Data quality checks completed."
        overall_score = 100

        total_checks = 0
        passed_checks = 0
        failed_checks = 0
        error_checks = 0

        if not (conn and database and schema and table):
            return (
                gr.DataFrame(visible=False), gr.DataFrame(visible=False),
                "❌ Please select database, schema, and table to run DQ checks.",
                "<div class='score-box failed-score'>Quality Score: N/A</div>",
                None, gr.Button(visible=False), None, gr.File(visible=False)
            )

        try:
            # Helper to process check results
            def process_check_result(res_or_list, penalty):
                nonlocal total_checks, passed_checks, failed_checks, error_checks, overall_score
                if isinstance(res_or_list, list): # For checks that return a list of results (e.g., column-specific checks)
                    for res in res_or_list:
                        # Only count and add to detailed_results if not a "Skip" status
                        if res.get("Status") != "⚠️ Skip":
                            total_checks += 1
                            detailed_results.append(res)
                            if res["Status"] == "✅ Pass":
                                passed_checks += 1
                            elif res["Status"] == "❌ Fail":
                                failed_checks += 1
                                overall_score -= penalty
                            elif res["Status"] == "❌ Error":
                                error_checks += 1
                                overall_score -= penalty
                elif res_or_list is not None and res_or_list.get("Status") != "⚠️ Skip": # For checks that return a single result dict
                    total_checks += 1
                    detailed_results.append(res_or_list)
                    if res_or_list["Status"] == "✅ Pass":
                        passed_checks += 1
                    elif res_or_list["Status"] == "❌ Fail":
                        failed_checks += 1
                        overall_score -= penalty
                    elif res_or_list["Status"] == "❌ Error":
                        error_checks += 1
                        overall_score -= penalty

            # Standard Table Checks
            if dq_check_row_count:
                res = self._run_row_count_check(database, schema, table, dq_min_rows)
                process_check_result(res, 10)

            if dq_check_duplicate_rows:
                res = self._run_duplicate_rows_check(database, schema, table)
                process_check_result(res, 10)

            # Column Null Percentage Check (always runs if enabled)
            if dq_check_column_null_pct:
                col_null_results = self._run_column_null_percentage_check(database, schema, table, dq_selected_columns_null, dq_null_threshold)
                process_check_result(col_null_results, 5)

            # Table Overall Null Percentage Check
            if dq_check_table_overall_null_pct:
                res = self._run_table_overall_null_percentage_check(database, schema, table, dq_table_null_threshold)
                process_check_result(res, 15) # Higher penalty for overall nulls

            # Column Checks (Conditional based on inputs)
            if dq_check_value_range:
                val_range_results = self._run_value_range_check(database, schema, table, dq_value_range_table.values.tolist())
                # Only process if the check was not skipped due to missing inputs
                if val_range_results: # _run_value_range_check returns [] if all rows have no min/max
                    process_check_result(val_range_results, 5)

            if dq_check_date_range:
                date_range_results = self._run_date_range_check(database, schema, table, dq_date_range_table.values.tolist())
                # Only process if the check was not skipped due to missing inputs
                if date_range_results: # _run_date_range_check returns [] if all rows have no min/max
                    process_check_result(date_range_results, 5)

            if dq_check_regex_pattern:
                regex_results = self._run_regex_pattern_check(database, schema, table, dq_selected_columns_regex, dq_pattern)
                # Only process if the check was not skipped due to missing pattern or columns
                if regex_results: # _run_regex_pattern_check returns [] if pattern or selected_columns are missing
                    process_check_result(regex_results, 5)

            # Relationship Checks
            if dq_check_fk:
                res = self._run_foreign_key_check(database, schema, table, dq_fk_column, dq_fk_ref_table, dq_fk_ref_column)
                # Only process if the check was not skipped due to incomplete config (res is None)
                if res is not None:
                    process_check_result(res, 15) # Higher penalty for FK issues

            # Ensure overall score doesn't go below zero
            overall_score = max(0, overall_score)

            # Create DataFrames for display
            # Summary DataFrame
            summary_data = [
                {"Metric": "Table", "Value": f"{database}.{schema}.{table}"},
                {"Metric": "Total Checks", "Value": total_checks},
                {"Metric": "Passed Checks", "Value": passed_checks},
                {"Metric": "Failed Checks", "Value": failed_checks},
                {"Metric": "Error Checks", "Value": error_checks},
                {"Metric": "Quality Score", "Value": f"{overall_score:.2f}%"}
            ]
            summary_df = pd.DataFrame(summary_data)

            # Detailed DataFrame
            detailed_df = pd.DataFrame(detailed_results, columns=["Check", "Column Name", "Expected", "Actual", "Status", "Details"])


            # Generate plot for overall score
            score_color = "green" if overall_score >= 80 else ("orange" if overall_score >= 50 else "red")
            score_class = "passed-score" if overall_score >= 80 else ("warning-score" if overall_score >= 50 else "failed-score")
            score_html = f"<div class='score-box {score_class}'>Quality Score: {overall_score:.0f}/100</div>"

            # Create a simple bar plot for summary statuses
            fig, ax = plt.subplots(figsize=(8, 5))
            if total_checks > 0:
                status_counts_data = {
                    "Passed": passed_checks,
                    "Failed": failed_checks,
                    "Error": error_checks,
                    # Skipped checks are not included in total_checks, so no need to subtract here
                }
                status_counts_series = pd.Series(status_counts_data)
                status_counts_series = status_counts_series[status_counts_series > 0] # Only show statuses with counts > 0

                if not status_counts_series.empty:
                    colors = {'Passed': 'green', 'Failed': 'red', 'Error': 'gray'} # No 'Skipped' color needed as they are not plotted
                    # Filter colors to only include those present in status_counts_series
                    plot_colors = [colors[status] for status in status_counts_series.index if status in colors]
                    status_counts_series.plot(kind='bar', ax=ax, color=plot_colors)
                    ax.set_title('Data Quality Check Status Summary')
                    ax.set_ylabel('Number of Checks')
                    ax.set_xlabel('Status')
                    plt.xticks(rotation=45, ha='right')
                else:
                    ax.text(0.5, 0.5, "No checks performed or no results.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    ax.set_title('Data Quality Check Status Summary')
                    ax.set_xticks([])
                    ax.set_yticks([])
            else:
                ax.text(0.5, 0.5, "No checks performed or no results.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.set_title('Data Quality Check Status Summary')
                ax.set_xticks([])
                ax.set_yticks([])

            plt.tight_layout()

            # Generate report file path
            report_filename = self._generate_dq_report(database, schema, table, summary_df, detailed_df)

            return (
                gr.DataFrame(value=summary_df, visible=True),
                gr.DataFrame(value=detailed_df, visible=True),
                status_message,
                score_html,
                fig, # Return the plot
                gr.Button(visible=True if report_filename else False), # Download button
                report_filename, # State holding filename
                gr.File(value=report_filename, visible=True if report_filename else False) # File component
            )

        except Exception as e:
            logging.error(f"An unhandled error occurred during DQ checks: {str(e)}")
            traceback.print_exc()
            return (
                gr.DataFrame(visible=False), gr.DataFrame(visible=False),
                f"❌ An unexpected error occurred: {str(e)}",
                "<div class='score-box failed-score'>Quality Score: N/A</div>",
                None, gr.Button(visible=False), None, gr.File(visible=False)
            )

    def _generate_dq_report(self, database, schema, table, summary_df, detailed_df):
        """Generates an Excel report for DQ results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("reports", exist_ok=True)
        excel_report_name = os.path.join("reports", f"dq_report_{database}_{schema}_{table}_{timestamp}.xlsx")
        try:
            with pd.ExcelWriter(excel_report_name, engine='openpyxl') as writer:
                # Convert timezone-aware datetimes to timezone-unaware before writing
                _make_datetimes_timezone_unaware(summary_df).to_excel(writer, sheet_name='DQ Summary', index=False)
                _make_datetimes_timezone_unaware(detailed_df).to_excel(writer, sheet_name='DQ Details', index=False)
            return excel_report_name
        except Exception as e:
            logging.error(f"Error generating DQ report: {str(e)}")
            traceback.print_exc()
            return None

def _make_datetimes_timezone_unaware(df):
    """Converts all timezone-aware datetime columns in a DataFrame to timezone-unaware."""
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=['datetime64[ns, UTC]', 'datetime64[ns]']).columns:
        if pd.api.types.is_datetime64_ns_dtype(df_copy[col]) and df_copy[col].dt.tz is not None:
            df_copy[col] = df_copy[col].dt.tz_localize(None)
    return df_copy


# ===== GRADIO APP =====
with gr.Blocks(title="DeploySure Suite", theme=gr.themes.Soft(), css="""
/* Base Styles */
body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background-color: #f9fafb;
    color: #2c3e50;
    margin: 0;
    padding: 0;
    min-height: 100%;
    display: flex;
    flex-direction: column;
}

/* Force font globally without altering color or layout */
*, *::before, *::after {
    font-family: 'Segoe UI', Arial, sans-serif !important;
}

/* Headings */
h1, h2, h3, h4 {
    font-weight: 600;
    color: #2c3e50;
    text-align: center;
}

/* Links */
a {
    color: #2563eb;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

/* Labels */
.label, span.svelte-g2oxp3 {
    display: inline-block;
    font-weight: 600;
    font-size: 14px;
    color: #2c3e50;
    margin-bottom: 4px;
    background-color: #e5e7eb;
    padding: 2px 6px;
    border-radius: 4px;
}

/* Input Fields */
.input, .textbox, .dropdown, .snowflake-input {
    width: 100%;
    padding: 12px;
    margin-bottom: 15px;
    border: 1px solid #ccc;
    border-radius: 6px;
    font-size: 14px;
    box-sizing: border-box;
    background-color: white;
}

/* Compact Login Form */
.snowflake-input {
    width: 100%;
    padding: 8px 12px;
    margin-bottom: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 14px;
}

.snowflake-button {
    padding: 8px 12px;
    margin: 5px;
    font-size: 14px;
}

/* Center the login form */
.gradio-container .tab {
    display: flex;
    justify-content: center;
}

/* Make the form container more compact */
.gr-form {
    max-width: 320px;
    padding: 15px;
}

/* Adjust heading size */
h2 {
    font-size: 18px !important;
    margin-bottom: 15px !important;
}

.snowflake-input:focus {
    border-color: #2563eb;
    outline: none;
    box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.1);
}

/* Buttons */
.button, .snowflake-button {
    width: 100%;
    padding: 10px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 15px;
    border: none;
    cursor: pointer;
    transition: background 0.3s ease;
}

.snowflake-button.primary {
    background-color: #2563eb;
    color: white;
}

.snowflake-button.primary:hover {
    background-color: #1d4ed8;
}

.snowflake-button.secondary {
    background-color: white;
    color: #2563eb;
    border: 1px solid #2563eb;
}

/* Status Box */
.status-box {
    background-color: #f8f9fa;
    padding: 12px;
    border-radius: 6px;
    font-size: 14px;
    margin-top: 10px;
    border-left: 4px solid #4CAF50;
    border: 1px solid #ddd;
}

.status-box.error {
    border-left-color: #f44336 !important;
}

/* Form Container */
.gr-form {
    max-width: 420px;
    margin: 0 auto;
    padding: 20px;
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
}

/* Tabs and Layouts */
.tab, .gr-tabs {
    padding: 20px;
    margin-top: 20px;
}

.gr-group {
    margin-bottom: 20px;
}

/* Dataframe Styling */
.gr-dataframe {
    max-height: 500px;
    overflow-y: auto;
}

/* Tab Selection Override */
.selected.svelte-1tcem6n.svelte-1tcem6n {
    background-color: transparent;
    color: #2563eb !important;
    font-weight: 600;
    padding: 8px 12px;
    border-bottom: 2px solid #2563eb;
}

/* Button Overrides */
.primary.svelte-1ixn6qd {
    border: var(--button-border-width) solid #2563eb;
    background: #2563eb;
    color: var(--button-primary-text-color);
    box-shadow: var(--button-primary-shadow);
}

button.svelte-1ixn6qd, a.svelte-1ixn6qd {
    display: inline-flex;
    justify-content: center;
    align-items: center;
    transition: var(--button-transition);
    padding: var(--size-0-5) var(--size-2);
    text-align: center;
}

/* Full Width Elements */
div.svelte-vt1mxs>*, div.svelte-vt1mxs>.form>* {
    width: var(--size-full);
}

/* Gradio Button Defaults */
.gradio-container button, .gradio-container [role=button] {
    cursor: pointer;
}

/* Box Model Fix */
.gradio-container, .gradio-container *, .gradio-container :before, .gradio-container :after {
    box-sizing: border-box;
    border-width: 0;
    border-style: solid;
}

/* Theme Variables */
:root {
    --name: default;
    --primary-500: #2563eb !important;
    --secondary-500: #3b82f6;
    --secondary-600: #2563eb;
    --neutral-100: #f4f4f5;
    --neutral-300: #d4d4d8;
    --neutral-700: #3f3f46;
    --color-accent-copied: #2563eb !important;
    --spacing-xxl: 16px;
    --radius-lg: 8px;
    --text-xs: 10px;
    --bg: white;
    --col: #27272a;
    --bg-dark: #0f0f11;
    --col-dark: #f4f4f5;
}

/* Test Case Checkbox Styling */
.gr-checkbox-group .gr-checkbox-item {
    background-color: #f5f5f5;
    border-radius: 4px;
    padding: 8px 12px;
    margin: 4px 0;
}

.gr-checkbox-group .gr-checkbox-item.selected {
    background-color: #e0e0e0 !important;
    border-left: 3px solid #555 !important;
}

.gr-checkbox-group .gr-checkbox-item:hover {
    background-color: #ebebeb;
}

.passed-score {
    background-color: #e6ffe6;
    border: 1px solid #00cc00;
    padding: 5px;
    border-radius: 5px;
    text-align: center;
    font-weight: bold;
}

.warning-score {
    background-color: #fffacd;
    border: 1px solid #ffcc00;
    padding: 5px;
    border-radius: 5px;
    text-align: center;
    font-weight: bold;
}

.failed-score {
    background-color: #ffe6e6;
    border: 1px solid #cc0000;
    padding: 5px;
    border-radius: 5px;
    text-align: center;
    font-weight: bold;
}

.score-box {
    margin-top: 10px;
}
""") as app:
    # Add company logo and header
    gr.HTML("""
    <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 10px; height: 10; width: 50;">
        <img src="data:image/png;base64,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">
        <h1 style="text-align: center;">DeploySure Suite </h1>
    </div>
    """)
    conn_state = gr.State(None)
    current_db = gr.State()
    combined_report_state = gr.State()
    validation_type = gr.State(value="schema")
    test_case_data = gr.State([])
    is_logged_in = gr.State(False)
    dq_report_filename_state = gr.State(None) # State to hold the single Excel filename
    kpi_report_filename_state = gr.State(None) # State to hold KPI Excel filename
    test_report_filename_state = gr.State(None) # State to hold Test Excel filename

    # ===== SNOWFLAKE-STYLE LOGIN SECTION =====
    with gr.Tab("🔐 Login"):
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("""
                <div style="text-align: center; margin-bottom: 15px;">
                    <h2 style="margin: 0; font-size: 18px;">Sign in to Snowflake</h2>
                </div>
                """)
                with gr.Group():
                    user = gr.Textbox(
                        label="Username",
                        placeholder="your_username",
                        elem_classes=["snowflake-input"]
                    )
                    password = gr.Textbox(
                        label="Password",
                        type="password",
                        placeholder="••••••••",
                        elem_classes=["snowflake-input"]
                    )
                    account = gr.Textbox(
                        label="Account",
                        placeholder="account.region",
                        elem_classes=["snowflake-input"]
                    )
                    with gr.Row():
                        login_btn = gr.Button(
                            "Connect",
                            variant="primary",
                            elem_classes=["snowflake-button"],
                            scale=1
                        )
                        disconnect_btn = gr.Button(
                            "Disconnect",
                            variant="secondary",
                            visible=False,
                            elem_classes=["snowflake-button"],
                            scale=1
                        )
                    status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        visible=False,
                        container=False,
                        elem_classes=["status-box"]
                    )

    # ===== MIRROR SCHEMA TAB =====
    with gr.Tab("⎘ MirrorSchema", visible=False) as mirror_tab:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Source Selection")
                source_db = gr.Dropdown(label="Source Database", interactive=True)
                source_schema = gr.Dropdown(label="Source Schema", interactive=True)
                target_schema = gr.Textbox(label="MirrorSchema Name", interactive=True, placeholder="Enter MirrorSchema name")
                clone_btn = gr.Button("Execute MirrorSchema", variant="primary")

            with gr.Column(scale=2):
                clone_output = gr.Textbox(label="Status", interactive=False)

    # ===== DRIFTWATCH TAB =====
    with gr.Tab("🔍 DriftWatch", visible=False) as driftwatch_tab:
        validation_type_dropdown = gr.Dropdown(
            label="Validation Type",
            choices=["Schema Validation", "KPI Validation", "Test Case Validation", "Data Quality Validation"],
            value="Schema Validation",
            interactive=True
        )

        with gr.Column(visible=True) as schema_validation_section:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Schema Validation Configuration")
                    val_db = gr.Dropdown(label="Database")
                    val_source_schema = gr.Dropdown(label="Source Schema")
                    val_target_schema = gr.Dropdown(label="Target Schema")
                    validate_btn = gr.Button("Execute DriftWatch", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("### ChangeLens / Schema Validation Report")
                    with gr.Tabs():
                        with gr.Tab("Table Differences"):
                            table_diff_output = gr.Dataframe(interactive=False)
                            table_download_btn = gr.Button("📥 Download Table Differences", visible=False)
                            table_download = gr.File(label="Download Table Differences", visible=False)
                        with gr.Tab("Column Differences"):
                            column_diff_output = gr.Dataframe(interactive=False)
                            column_download_btn = gr.Button("📥 Download Column Differences", visible=False)
                            column_download = gr.File(label="Download Column Differences", visible=False)
                        with gr.Tab("Data Type Differences"):
                            datatype_diff_output = gr.Dataframe(interactive=False)
                            datatype_download_btn = gr.Button("📥 Download Data Type Differences", visible=False)
                            datatype_download = gr.File(label="Download Data Type Differences", visible=False)

                    val_status = gr.Textbox(label="Status", interactive=False)
                    schema_download_btn = gr.Button("📥 Download Schema Validation Report", visible=False)
                    schema_download = gr.File(label="Download Schema Validation Report", visible=False)

        with gr.Column(visible=False) as kpi_validation_section:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### KPI Validation Configuration")
                    kpi_db = gr.Dropdown(label="Database")
                    kpi_source_schema = gr.Dropdown(label="Source Schema")
                    kpi_target_schema = gr.Dropdown(label="Target Schema")

                    with gr.Group():
                        gr.Markdown("### Select KPIs to Validate")
                        kpi_select_all = gr.Checkbox(label="Select All", value=True)
                        with gr.Row():
                            kpi_total_orders = gr.Checkbox(label="Total Orders", value=True)
                            kpi_total_revenue = gr.Checkbox(label="Total Revenue", value=True)
                            kpi_avg_order = gr.Checkbox(label="Average Order Value", value=True)
                        with gr.Row():
                            kpi_max_order = gr.Checkbox(label="Maximum Order Value", value=True)
                            kpi_min_order = gr.Checkbox(label="Minimum Order Value", value=True)
                            kpi_completed = gr.Checkbox(label="Completed Orders", value=True)
                        with gr.Row():
                            kpi_cancelled = gr.Checkbox(label="Cancelled Orders", value=True)
                            kpi_april_orders = gr.Checkbox(label="Orders in April 2025", value=True)
                            kpi_unique_customers = gr.Checkbox(label="Unique Customers", value=True)

                    kpi_validate_btn = gr.Button("Execute DriftWatch", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("### ChangeLens / KPI Validation Report")
                    kpi_output = gr.Dataframe(
                        interactive=False,
                        wrap=True
                    )
                    kpi_status = gr.Textbox(label="Status", interactive=False)
                    kpi_download_btn = gr.Button("📥 Download KPI Validation Report", visible=False)
                    kpi_download = gr.File(label="Download KPI Validation Report", visible=False)

        with gr.Column(visible=False) as test_case_validation_section:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Test Automation")
                    tc_db = gr.Dropdown(label="Database")
                    tc_schema = gr.Dropdown(label="Schema")
                    tc_table = gr.Dropdown(
                        label="Catagory",
                        choices=["All"],
                        value="All",
                        allow_custom_value=True
                    )

                    with gr.Group():
                        gr.Markdown("### Select Test Cases")
                        tc_select_all = gr.Checkbox(
                            label="Select All",
                            value=True,
                            info="Toggle all test cases for the selected table",
                        )
                        tc_test_cases = gr.CheckboxGroup(
                            label="Available Test Cases",
                            choices=[],
                            interactive=True
                        )

                    tc_validate_btn = gr.Button("Execute DriftWatch", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("### ChangeLens / Test Automation Report")
                    tc_output = gr.Dataframe(
                        interactive=False,
                        wrap=True
                    )
                    tc_status = gr.Textbox(label="Status", interactive=False)
                    tc_download_btn = gr.Button("📥 Download Test Report", visible=False)
                    tc_download = gr.File(label="Download Test Report", visible=False)

        # ===== DATA QUALITY VALIDATION SECTION =====
        with gr.Column(visible=False) as data_quality_section:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Data Quality Configuration")
                    dq_db = gr.Dropdown(label="Database", choices=[], interactive=True)
                    dq_schema = gr.Dropdown(label="Schema", choices=[], interactive=True)
                    dq_table = gr.Dropdown(label="Table", choices=[], interactive=True)

                    # Standard Table Checks
                    gr.Markdown("### Standard Table Checks")
                    dq_check_row_count = gr.Checkbox(label="Row Count", value=True)
                    dq_min_rows = gr.Number(label="Minimum Expected Rows", value=1, visible=True)
                    dq_check_duplicate_rows = gr.Checkbox(label="Duplicate Rows", value=True)

                    # Column Null Percentage Check (now under Standard Checks)
                    with gr.Group():
                        dq_check_column_null_pct = gr.Checkbox(label="Column Null Percentage Check", value=False)
                        dq_selected_columns_null = gr.Dropdown(
                            label="Select Columns for Null Percentage Check (Leave empty to check all applicable columns)", # Updated label
                            choices=[],
                            multiselect=True,
                            interactive=True,
                            visible=False, # Controlled by checkbox
                            value=[] # Ensure it's empty when hidden
                        )
                        dq_null_threshold = gr.Slider(
                            label="Null % Threshold",
                            value=10,
                            minimum=0,
                            maximum=100,
                            interactive=True,
                            visible=False # Controlled by checkbox
                        )

                    # New: Table Overall Null Percentage Check
                    with gr.Group():
                        dq_check_table_overall_null_pct = gr.Checkbox(label="Table Overall Null Percentage Check", value=False)
                        dq_table_null_threshold = gr.Slider(
                            label="Overall Null % Threshold",
                            value=5,
                            minimum=0,
                            maximum=100,
                            interactive=True,
                            visible=False # Controlled by checkbox
                        )

                    # Column Checks (Remaining)
                    gr.Markdown("### Column Checks")
                    # Value Range Check
                    with gr.Group():
                        dq_check_value_range = gr.Checkbox(label="Value Range Check", value=False)
                        dq_value_range_table = gr.Dataframe(
                            headers=["Column", "Min Value", "Max Value"],
                            datatype=["str", "str", "str"], # Changed to str to avoid Gradio's automatic type conversion issues with empty cells
                            value=[["", "", ""]], # Default empty row, use empty strings for numbers
                            row_count="dynamic",
                            col_count=3,
                            interactive=True,
                            visible=False, # Controlled by checkbox
                            label="Configure Column-specific Value Ranges"
                        )

                    # Date Range Check
                    with gr.Group():
                        dq_check_date_range = gr.Checkbox(label="Date Range Check", value=False)
                        dq_date_range_table = gr.Dataframe(
                            headers=["Column", "Min Date", "Max Date"],
                            datatype=["str", "str", "str"],
                            value=[["", "2000-01-01", datetime.now().strftime("%Y-%m-%d")]],
                            row_count="dynamic",
                            col_count=3,
                            interactive=True,
                            visible=False,
                            label="Configure Column-specific Date Ranges"
                        )

                    # Regex Pattern Check
                    with gr.Group():
                        dq_check_regex_pattern = gr.Checkbox(label="Regex Pattern Check", value=False)
                        dq_selected_columns_regex = gr.Dropdown(
                            label="Select Columns for Regex Check (Leave empty to check all string columns)", # Updated label
                            choices=[],
                            multiselect=True,
                            interactive=True,
                            visible=False # Controlled by checkbox
                        )
                        dq_pattern = gr.Textbox(
                            label="Regex Pattern",
                            value="^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}$",
                            visible=False # Controlled by checkbox
                        )

                    # Relationship Checks
                    gr.Markdown("### Relationship Checks")
                    dq_check_fk = gr.Checkbox(label="Foreign Key Check", value=False)
                    dq_fk_column = gr.Dropdown(
                        label="Foreign Key Column (Current Table)",
                        choices=[],
                        interactive=True,
                        value=None, # Set initial value to None
                        visible=False # Controlled by checkbox
                    )
                    dq_fk_ref_table = gr.Dropdown(
                        label="Referenced Table",
                        choices=[],
                        interactive=True,
                        value=None, # Set initial value to None
                        visible=False # Controlled by checkbox
                    )
                    dq_fk_ref_column = gr.Dropdown(
                        label="Referenced Column",
                        choices=[],
                        interactive=True,
                        value=None, # Set initial value to None
                        visible=False # Controlled by checkbox
                    )
                    dq_run_btn = gr.Button("Run Data Quality Checks", variant="primary")
                    with gr.Row(visible=False) as dq_loading:
                        gr.HTML("<div style='text-align: center;'><i>Running checks... This may take a few moments.</i></div>")
                with gr.Column(scale=2):
                    gr.Markdown("### Data Quality Results")
                    with gr.Tabs():
                        with gr.Tab("Summary"):
                            dq_summary = gr.Dataframe(
                                headers=["Metric", "Value"],
                                interactive=False,
                                wrap=True,
                                visible=False # Initially hidden
                            )
                            dq_score = gr.HTML(
                                value="<div class='score-box'>Quality Score: N/A</div>",
                                label="Quality Score"
                            )
                            dq_plot = gr.Plot(visible=False) # Initially hidden
                        with gr.Tab("Detailed Results"):
                            dq_details = gr.Dataframe(
                                headers=["Check", "Column Name", "Expected", "Actual", "Status", "Details"],
                                interactive=False,
                                wrap=True,
                                visible=False # Initially hidden
                            )
                    dq_status = gr.Textbox(label="Execution Status", interactive=False)
                    dq_download_report_button = gr.Button("📥 Download Data Quality Report", visible=False)
                    # This gr.File component will be updated by the download button click
                    dq_download_file = gr.File(label="Download Data Quality Report", visible=False)
                    # The dq_report_filename_state will hold the actual path for download
                    # It's already defined at the top: dq_report_filename_state = gr.State(None)


    # ===== EVENT HANDLERS =====
    def toggle_validation_type(validation_type_selection):
        schema_vis = (validation_type_selection == "Schema Validation")
        kpi_vis = (validation_type_selection == "KPI Validation")
        test_case_vis = (validation_type_selection == "Test Case Validation")
        dq_vis = (validation_type_selection == "Data Quality Validation")

        return (
            gr.Column(visible=schema_vis),
            gr.Column(visible=kpi_vis),
            gr.Column(visible=test_case_vis),
            gr.Column(visible=dq_vis),
            validation_type_selection
        )

    validation_type_dropdown.change(
        toggle_validation_type,
        inputs=validation_type_dropdown,
        outputs=[schema_validation_section, kpi_validation_section, test_case_validation_section, data_quality_section, validation_type])

    def handle_login(user, password, account):
        conn, msg = get_snowflake_connection(user, password, account)
        success = conn is not None
        return (
            conn,
            msg,
            success,
            gr.Tab(visible=success),
            gr.Tab(visible=success),
            gr.Button(visible=success),
            gr.Button(visible=not success),
            gr.Textbox(visible=True)
        )

    def handle_logout(conn):
        conn, msg = disconnect_snowflake(conn)
        return (
            conn,
            msg,
            False,
            gr.Tab(visible=False),
            gr.Tab(visible=False),
            gr.Button(visible=False),
            gr.Button(visible=True),
            gr.Textbox(visible=True)
        )

    login_btn.click(
        handle_login,
        inputs=[user, password, account],
        outputs=[conn_state, status, is_logged_in, mirror_tab, driftwatch_tab, disconnect_btn, login_btn, status]
    )

    disconnect_btn.click(
        handle_logout,
        inputs=[conn_state],
        outputs=[conn_state, status, is_logged_in, mirror_tab, driftwatch_tab, disconnect_btn, login_btn, status]
    )

    def update_schemas(conn, db, source_schema_val):
        if conn and db:
            schemas = get_schemas(conn, db)
            current_source_schema_val = source_schema_val if source_schema_val in schemas else (schemas[0] if schemas else None)
            suggested_name = f"{current_source_schema_val}_CLONE" if current_source_schema_val else ""
            return (
                gr.Dropdown(choices=schemas, value=current_source_schema_val, interactive=True),
                gr.Textbox(value=suggested_name, interactive=True)
            )
        return (
            gr.Dropdown(interactive=False, value=None),
            gr.Textbox(interactive=False, value="")
        )

    def init_mirror_ui(conn):
        if conn:
            dbs = get_databases(conn)
            return (
                gr.Dropdown(choices=dbs, value=dbs[0] if dbs else None, interactive=True),
                gr.Dropdown(interactive=False, value=None),
            )
        return (
            gr.Dropdown(interactive=False, value=None),
            gr.Dropdown(interactive=False, value=None),
        )

    source_db.change(
        update_schemas,
        inputs=[conn_state, source_db, source_schema],
        outputs=[source_schema, target_schema]
    )

    source_schema.change(
        update_schemas,
        inputs=[conn_state, source_db, source_schema],
        outputs=[source_schema, target_schema]
    )

    is_logged_in.change(
        init_mirror_ui,
        inputs=[conn_state],
        outputs=[source_db, source_schema],
        queue=False
    )

    clone_btn.click(
        execute_clone,
        inputs=[conn_state, source_db, source_schema, target_schema],
        outputs=[clone_output]
    )

    def update_val_schemas(conn, db):
        if conn and db:
            schemas = get_schemas(conn, db)
            return (
                gr.Dropdown(choices=schemas, value=schemas[0] if schemas else None),
                gr.Dropdown(choices=schemas, value=schemas[0] if schemas else None)
            )
        return gr.Dropdown(value=None), gr.Dropdown(value=None)

    def init_validation_ui(conn):
        if conn:
            dbs = get_databases(conn)
            return (
                gr.Dropdown(choices=dbs, value=dbs[0] if dbs else None),
                gr.Dropdown(value=None),
                gr.Dropdown(value=None),
            )
        return (
            gr.Dropdown(value=None),
            gr.Dropdown(value=None),
            gr.Dropdown(value=None),
        )

    val_db.change(
        update_val_schemas,
        inputs=[conn_state, val_db],
        outputs=[val_source_schema, val_target_schema],
        queue=False
    )

    is_logged_in.change(
        init_validation_ui,
        inputs=[conn_state],
        outputs=[val_db, val_source_schema, val_target_schema],
        queue=False
    )

    def run_validation(conn, db, source_schema, target_schema):
        try:
            table_diff = compare_table_differences(conn, db, source_schema, target_schema)
            column_diff, datatype_diff = compare_column_differences(conn, db, source_schema, target_schema)

            all_cols = list(set(table_diff.columns).union(column_diff.columns).union(datatype_diff.columns))

            table_diff_reindexed = table_diff.reindex(columns=all_cols, fill_value=None)
            column_diff_reindexed = column_diff.reindex(columns=all_cols, fill_value=None)
            datatype_diff_reindexed = datatype_diff.reindex(columns=all_cols, fill_value=None)

            combined_df = pd.concat([
                table_diff_reindexed.assign(Validation_Type="Table Differences"),
                column_diff_reindexed.assign(Validation_Type="Column Differences"),
                datatype_diff_reindexed.assign(Validation_Type="Data Type Differences")
            ], ignore_index=True)

            return (
                gr.Dataframe(value=table_diff, visible=True if not table_diff.empty else False),
                gr.Dataframe(value=column_diff, visible=True if not column_diff.empty else False),
                gr.Dataframe(value=datatype_diff, visible=True if not datatype_diff.empty else False),
                "✅ Validation completed successfully!",
                gr.Button(visible=not table_diff.empty),
                gr.Button(visible=not column_diff.empty),
                gr.Button(visible=not datatype_diff.empty),
                combined_df,
                gr.Button(visible=not combined_df.empty)
            )
        except Exception as e:
            logging.error(f"Validation failed: {str(e)}")
            traceback.print_exc()
            return (
                gr.Dataframe(visible=False),
                gr.Dataframe(visible=False),
                gr.Dataframe(visible=False),
                f"❌ Validation failed: {str(e)}",
                gr.Button(visible=False),
                gr.Button(visible=False),
                gr.Button(visible=False),
                pd.DataFrame(),
                gr.Button(visible=False)
            )

    validate_btn.click(
        run_validation,
        inputs=[conn_state, val_db, val_source_schema, val_target_schema],
        outputs=[
            table_diff_output,
            column_diff_output,
            datatype_diff_output,
            val_status,
            table_download_btn,
            column_download_btn,
            datatype_download_btn,
            combined_report_state,
            schema_download_btn
        ]
    )

    def download_table_report(df):
        if df.empty:
            return None, gr.File(visible=False)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Table_Differences_{timestamp}.csv"
        df.to_csv(filename, index=False)
        return filename, gr.File(value=filename, visible=True, label="Download Table Differences")

    def download_column_report(df):
        if df.empty:
            return None, gr.File(visible=False)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Column_Differences_{timestamp}.csv"
        df.to_csv(filename, index=False)
        return filename, gr.File(value=filename, visible=True, label="Download Column Differences")

    def download_datatype_report(df):
        if df.empty:
            return None, gr.File(visible=False)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Datatype_Differences_{timestamp}.csv"
        df.to_csv(filename, index=False)
        return filename, gr.File(value=filename, visible=True, label="Download Data Type Differences")

    def download_schema_report(combined_df):
        if combined_df.empty:
            return None, gr.File(visible=False)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Schema_Validation_Report_{timestamp}.csv"
        combined_df.to_csv(filename, index=False)
        return filename, gr.File(value=filename, visible=True, label="Download Schema Validation Report")

    table_download_btn.click(
        download_table_report,
        inputs=table_diff_output,
        outputs=[table_download, table_download]
    )

    column_download_btn.click(
        download_column_report,
        inputs=column_diff_output,
        outputs=[column_download, column_download]
    )

    datatype_download_btn.click(
        download_datatype_report,
        inputs=datatype_diff_output,
        outputs=[datatype_download, datatype_download]
    )

    schema_download_btn.click(
        download_schema_report,
        inputs=combined_report_state,
        outputs=[schema_download, schema_download]
    )

    def update_kpi_schemas(conn, db):
        if conn and db:
            schemas = get_schemas(conn, db)
            return (
                gr.Dropdown(choices=schemas, value=schemas[0] if schemas else None),
                gr.Dropdown(choices=schemas, value=schemas[0] if schemas else None)
            )
        return gr.Dropdown(value=None), gr.Dropdown(value=None)

    def init_kpi_ui(conn):
        if conn:
            dbs = get_databases(conn)
            return (
                gr.Dropdown(choices=dbs, value=dbs[0] if dbs else None),
                gr.Dropdown(value=None),
                gr.Dropdown(value=None),
            )
        return (
            gr.Dropdown(value=None),
            gr.Dropdown(value=None),
            gr.Dropdown(value=None),
        )

    kpi_db.change(
        update_kpi_schemas,
        inputs=[conn_state, kpi_db],
        outputs=[kpi_source_schema, kpi_target_schema],
        queue=False
    )

    is_logged_in.change(
        init_kpi_ui,
        inputs=[conn_state],
        outputs=[kpi_db, kpi_source_schema, kpi_target_schema],
        queue=False
    )

    def toggle_all_kpis(select_all):
        return (
            gr.Checkbox(value=select_all),
            gr.Checkbox(value=select_all),
            gr.Checkbox(value=select_all),
            gr.Checkbox(value=select_all),
            gr.Checkbox(value=select_all),
            gr.Checkbox(value=select_all),
            gr.Checkbox(value=select_all),
            gr.Checkbox(value=select_all),
            gr.Checkbox(value=select_all)
        )

    kpi_select_all.change(
        toggle_all_kpis,
        inputs=kpi_select_all,
        outputs=[
            kpi_total_orders, kpi_total_revenue, kpi_avg_order,
            kpi_max_order, kpi_min_order, kpi_completed,
            kpi_cancelled, kpi_april_orders, kpi_unique_customers
        ]
    )

    def validate_selected_kpis(conn, database, source_schema, target_schema, *kpi_selections):
        cursor = conn.cursor()
        results = []

        kpi_mapping = {
            "Total Orders": 1,
            "Total Revenue": 2,
            "Average Order Value": 3,
            "Maximum Order Value": 4,
            "Minimum Order Value": 5,
            "Completed Orders": 6,
            "Cancelled Orders": 7,
            "Orders in April 2025": 8,
            "Unique Customers": 9
        }

        selected_kpi_names = [kpi_name for kpi_name, selected in zip(kpi_mapping.keys(), kpi_selections) if selected]

        if not selected_kpi_names:
            return pd.DataFrame(), "⚠️ No KPIs selected for validation", gr.Button(visible=False)

        try:
            kpi_query = f"""
            SELECT KPI_ID, KPI_NAME, KPI_VALUE
            FROM {database}.{source_schema}.ORDER_KPIS
            WHERE KPI_NAME IN ({','.join([f"'{kpi}'" for kpi in selected_kpi_names])})
            """
            cursor.execute(kpi_query)
            kpis = cursor.fetchall()

            if not kpis:
                return pd.DataFrame(), "⚠️ No matching KPIs found in ORDER_KPIS table", gr.Button(visible=False)

            source_has_table = False
            target_has_table = False
            try:
                cursor.execute(f"SELECT 1 FROM {database}.{source_schema}.ORDER_DATA LIMIT 1")
                source_has_table = True
            except Exception as e:
                logging.warning(f"ORDER_DATA table not found in source schema {source_schema}: {e}")
                traceback.print_exc()

            try:
                cursor.execute(f"SELECT 1 FROM {database}.{target_schema}.ORDER_DATA LIMIT 1")
                target_has_table = True
            except Exception as e:
                logging.warning(f"ORDER_DATA table not found in target schema {target_schema}: {e}")
                traceback.print_exc()

            if not source_has_table or not target_has_table:
                error_msg = "ORDER_DATA table missing in "
                if not source_has_table and not target_has_table:
                    error_msg += "both schemas"
                elif not source_has_table:
                    error_msg += "source schema"
                else:
                    error_msg += "target schema"

                for kpi_id, kpi_name, kpi_sql in kpis:
                    results.append({
                        'KPI ID': kpi_id,
                        'KPI Name': kpi_name,
                        'Source Value': f"ERROR: {error_msg}",
                        'Clone Value': f"ERROR: {error_msg}",
                        'Difference': "N/A",
                        'Diff %': "N/A",
                        'Status': "❌ Error"
                    })
                return pd.DataFrame(results), "❌ Validation failed - missing ORDER_DATA table", gr.Button(visible=False)

            for kpi_id, kpi_name, kpi_sql in kpis:
                result_source = "N/A"
                result_clone = "N/A"
                try:
                    source_query = re.sub(r'\bORDER_DATA\b', f'{database}.{source_schema}.ORDER_DATA', kpi_sql, flags=re.IGNORECASE)
                    cursor.execute(source_query)
                    result_source = cursor.fetchone()[0] if cursor.rowcount > 0 else None
                except Exception as e:
                    result_source = f"QUERY_ERROR: {str(e)}"
                    logging.error(f"Error executing source KPI query for {kpi_name}: {e}")
                    traceback.print_exc()

                try:
                    clone_query = re.sub(r'\bORDER_DATA\b', f'{database}.{target_schema}.ORDER_DATA', kpi_sql, flags=re.IGNORECASE)
                    cursor.execute(clone_query)
                    result_clone = cursor.fetchone()[0] if cursor.rowcount > 0 else None
                except Exception as e:
                    result_clone = f"QUERY_ERROR: {str(e)}"
                    logging.error(f"Error executing clone KPI query for {kpi_name}: {e}")
                    traceback.print_exc()

                diff = "N/A"
                pct_diff = "N/A"
                status = "⚠️ Mismatch"

                try:
                    if (isinstance(result_source, (int, float)) and isinstance(result_clone, (int, float))):
                        diff = float(result_source) - float(result_clone)
                        if float(result_source) != 0:
                            pct_diff = (diff / float(result_source)) * 100
                        else:
                            pct_diff = "N/A"
                        status = '✅ Match' if diff == 0 else '⚠️ Mismatch'
                    elif str(result_source) == str(result_clone):
                        status = '✅ Match'
                except Exception as e:
                    logging.warning(f"Error comparing KPI values for {kpi_name}: {e}. Setting diff/pct_diff to N/A.")
                    traceback.print_exc()
                    pass

                results.append({
                    'KPI ID': kpi_id,
                    'KPI Name': kpi_name,
                    'Source Value': result_source,
                    'Clone Value': result_clone,
                    'Difference': diff if not isinstance(diff, float) else round(diff, 2),
                    'Diff %': f"{round(pct_diff, 2)}%" if isinstance(pct_diff, float) else pct_diff,
                    'Status': status
                })

            df = pd.DataFrame(results)
            return gr.Dataframe(value=df, visible=True if not df.empty else False), "✅ KPI validation completed", gr.Button(visible=True)

        except Exception as e:
            logging.error(f"KPI validation failed: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame(), f"❌ KPI validation failed: {str(e)}", gr.Button(visible=False)

    kpi_validate_btn.click(
        validate_selected_kpis,
        inputs=[
            conn_state, kpi_db, kpi_source_schema, kpi_target_schema,
            kpi_total_orders, kpi_total_revenue, kpi_avg_order,
            kpi_max_order, kpi_min_order, kpi_completed,
            kpi_cancelled, kpi_april_orders, kpi_unique_customers
        ],
        outputs=[kpi_output, kpi_status, kpi_download_btn]
    )

    def download_kpi_report(df):
        if df.empty:
            return None, gr.File(visible=False)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"KPI_Validation_Report_{timestamp}.csv"
        df.to_csv(filename, index=False)
        return filename, gr.File(value=filename, visible=True, label="Download KPI Validation Report")

    kpi_download_btn.click(
        download_kpi_report,
        inputs=kpi_output,
        outputs=[kpi_download, kpi_download]
    )

    def init_test_case_ui(conn):
        if conn:
            dbs = get_databases(conn)
            return (
                gr.Dropdown(choices=dbs, value=dbs[0] if dbs else None),
                gr.Dropdown(value=None),
                gr.Dropdown(choices=["All"], value="All", allow_custom_value=True),
            )
        return (
            gr.Dropdown(value=None),
            gr.Dropdown(value=None),
            gr.Dropdown(choices=["All"], value="All", allow_custom_value=True),
        )

    is_logged_in.change(
        init_test_case_ui,
        inputs=[conn_state],
        outputs=[tc_db, tc_schema, tc_table],
        queue=False
    )

    def update_tc_schemas(conn, db):
        if conn and db:
            schemas = get_schemas(conn, db)
            return gr.Dropdown(choices=schemas, value=schemas[0] if schemas else None)
        return gr.Dropdown(value=None)

    tc_db.change(
        update_tc_schemas,
        inputs=[conn_state, tc_db],
        outputs=tc_schema,
        queue=False
    )

    def update_test_case_components(conn, db, schema, table, select_all):
        logging.info(f"Updating test case components for: {db}.{schema}.{table}")

        if not (conn and db and schema):
            logging.warning("Missing required parameters for test case component update.")
            return (
                gr.Dropdown(choices=["All"], value="All", allow_custom_value=True),
                gr.CheckboxGroup(choices=[]),
                [],
                gr.Checkbox(interactive=False) # Set to interactive=False if no choices
            )

        try:
            tables = get_test_case_tables(conn, db, schema)
            logging.info(f"Available test case tables: {tables}")

            test_cases = get_test_cases(conn, db, schema, table)
            logging.info(f"Found {len(test_cases)} test cases for {table}")

            choices = [f"{case[1]}" for case in test_cases]

            current_table_value = table if table in tables else "All"

            return (
                gr.Dropdown(choices=tables, value=current_table_value, allow_custom_value=True),
                gr.CheckboxGroup(
                    choices=choices,
                    value=choices if select_all else [],
                    label=f"Available Test Cases"
                ),
                test_cases,
                gr.Checkbox(value=select_all, interactive=len(choices) > 0)
            )

        except Exception as e:
            logging.error(f"Test case component update error: {str(e)}")
            traceback.print_exc()
            return (
                gr.Dropdown(choices=["All"], value="All", allow_custom_value=True),
                gr.CheckboxGroup(choices=[], label="Available Test Cases"),
                [],
                gr.Checkbox(interactive=False)
            )

    tc_schema.change(
        update_test_case_components,
        inputs=[conn_state, tc_db, tc_schema, tc_table, tc_select_all],
        outputs=[tc_table, tc_test_cases, test_case_data, tc_select_all],
        queue=False
    )

    tc_table.change(
        update_test_case_components,
        inputs=[conn_state, tc_db, tc_schema, tc_table, tc_select_all],
        outputs=[tc_table, tc_test_cases, test_case_data, tc_select_all],
        queue=False
    )

    def toggle_all_test_cases(select_all, test_case_data_state):
        """Toggle all test cases selection"""
        all_choices = [f"{case[1]}" for case in test_case_data_state]
        return (
            gr.CheckboxGroup(value=all_choices if select_all else []),
            select_all
        )

    tc_select_all.change(
        toggle_all_test_cases,
        inputs=[tc_select_all, test_case_data],
        outputs=[tc_test_cases, tc_select_all]
    )

    def execute_test_case_validation(conn, db, schema, selected_case_names, test_case_data):
        if not conn or not db or not schema:
            return pd.DataFrame(), "❌ Please select database and schema", gr.Button(visible=False)

        if not selected_case_names:
            return pd.DataFrame(), "⚠️ Please select at least one test case", gr.Button(visible=False)

        selected_test_cases = []
        for name in selected_case_names:
            for case in test_case_data:
                if case[1] == name:
                    selected_test_cases.append(case)
                    break

        if not selected_test_cases:
            return pd.DataFrame(), "⚠️ No valid test cases selected", gr.Button(visible=False)

        df, status_msg, btn_vis = validate_test_cases(conn, db, schema, selected_test_cases)
        return gr.Dataframe(value=df, visible=True if not df.empty else False), status_msg, btn_vis

    tc_validate_btn.click(
        execute_test_case_validation,
        inputs=[conn_state, tc_db, tc_schema, tc_test_cases, test_case_data],
        outputs=[tc_output, tc_status, tc_download_btn]
    )

    def download_test_case_report(df, test_case_data_state):
        if df.empty:
            return None, gr.File(visible=False)

        desc_map = {case[1]: case[3] for case in test_case_data_state}

        df['DESCRIPTION'] = df['TEST CASE'].map(desc_map)

        cols = ['TEST CASE', 'DESCRIPTION', 'CATEGORY', 'EXPECTED RESULT', 'ACTUAL RESULT', 'STATUS']
        df = df[cols]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Test_Case_Validation_Report_{timestamp}.csv"
        df.to_csv(filename, index=False)
        return filename, gr.File(value=filename, visible=True, label="Download Test Report")

    tc_download_btn.click(
        download_test_case_report,
        inputs=[tc_output, test_case_data],
        outputs=[tc_download, tc_download]
    )

    # Data Quality DB/Schema/Table changes
    def update_dq_schemas_dropdown(conn, db):
        if conn and db:
            schemas = get_schemas(conn, db)
            return gr.Dropdown(choices=schemas, value=schemas[0] if schemas else None)
        return gr.Dropdown(value=None, choices=[])

    def update_dq_tables_dropdown(conn, db, schema):
        if conn and db and schema:
            tables = get_tables(conn, db, schema)
            return gr.Dropdown(choices=tables, value=tables[0] if tables else None)
        return gr.Dropdown(value=None, choices=[])

    dq_db.change(
        update_dq_schemas_dropdown,
        inputs=[conn_state, dq_db],
        outputs=dq_schema
    )
    dq_schema.change(
        update_dq_tables_dropdown,
        inputs=[conn_state, dq_db, dq_schema],
        outputs=dq_table
    )

    # This is the crucial handler for populating DQ column dropdowns and dataframes
    dq_table.change(
        lambda conn, db, schema, table: (
            lambda col_details: (
                gr.Dropdown(choices=col_details[0]), # dq_selected_columns_null (all columns)
                gr.Dropdown(choices=col_details[0]), # dq_fk_column (all columns)
                gr.Dropdown(choices=get_tables(conn, db, schema)), # dq_fk_ref_table (all tables)
                gr.Dropdown(choices=col_details[3]), # dq_selected_columns_regex (string columns)
                gr.Dataframe(
                    value=[[col_name, "", ""] for col_name in col_details[1]] if col_details[1] else [["", "", ""]], # dq_value_range_table (numeric columns)
                    headers=["Column", "Min Value", "Max Value"],
                    datatype=["str", "str", "str"],
                    row_count="dynamic",
                    col_count=3,
                    interactive=True
                ),
                gr.Dataframe(
                    value=[[col_name, "2000-01-01", datetime.now().strftime("%Y-%m-%d")] for col_name in col_details[2]] if col_details[2] else [["", "2000-01-01", datetime.now().strftime("%Y-%m-%d")]], # dq_date_range_table (date columns)
                    headers=["Column", "Min Date", "Max Date"],
                    datatype=["str", "str", "str"],
                    row_count="dynamic",
                    col_count=3,
                    interactive=True
                )
            )
        )(_categorize_columns_by_type(_get_column_details_for_dq(conn, db, schema, table))), # Call _get_column_details_for_dq and _categorize_columns_by_type here
        inputs=[conn_state, dq_db, dq_schema, dq_table],
        outputs=[
            dq_selected_columns_null,
            dq_fk_column,
            dq_fk_ref_table,
            dq_selected_columns_regex,
            dq_value_range_table,
            dq_date_range_table
        ]
    )


    dq_fk_ref_table.change(
        lambda conn, db, schema, table: gr.Dropdown(
            choices=get_columns_for_table(conn, db, schema, table), # Use the simple get_columns_for_table here
            value=None,
            interactive=True
        ),
        inputs=[conn_state, dq_db, dq_schema, dq_fk_ref_table],
        outputs=dq_fk_ref_column
    )


    # Initialize DQ UI components on login
    def init_dq_ui(conn):
        if conn:
            dbs = get_databases(conn)
            return (
                gr.Dropdown(choices=dbs, value=dbs[0] if dbs else None),
                gr.Dropdown(value=None),
                gr.Dropdown(value=None)
            )
        return (
            gr.Dropdown(value=None),
            gr.Dropdown(value=None),
            gr.Dropdown(value=None)
        )

    is_logged_in.change(
        init_dq_ui,
        inputs=[conn_state],
        outputs=[dq_db, dq_schema, dq_table],
        queue=False
    )


    # ===== CHECKBOX VISIBILITY TOGGLES FOR DQ SECTION =====
    dq_check_column_null_pct.change(
        lambda x: [gr.Dropdown(visible=x), gr.Slider(visible=x)],
        inputs=dq_check_column_null_pct,
        outputs=[dq_selected_columns_null, dq_null_threshold]
    )
    dq_check_table_overall_null_pct.change(
        lambda x: gr.Slider(visible=x),
        inputs=dq_check_table_overall_null_pct,
        outputs=dq_table_null_threshold
    )
    dq_check_value_range.change(
        lambda x: gr.Dataframe(visible=x), # Only toggle visibility of the dataframe
        inputs=dq_check_value_range,
        outputs=dq_value_range_table
    )
    dq_check_date_range.change(
        lambda x: gr.Dataframe(visible=x),
        inputs=dq_check_date_range,
        outputs=dq_date_range_table
    )
    dq_check_regex_pattern.change(
        lambda x: [gr.Dropdown(visible=x), gr.Textbox(visible=x)],
        inputs=dq_check_regex_pattern,
        outputs=[dq_selected_columns_regex, dq_pattern]
    )
    dq_check_fk.change(
        lambda x: [gr.Dropdown(visible=x), gr.Dropdown(visible=x), gr.Dropdown(visible=x)],
        inputs=dq_check_fk,
        outputs=[dq_fk_column, dq_fk_ref_table, dq_fk_ref_column]
    )

    # ===== CORE FUNCTION CALLS =====

    # Data Quality Run Button
    dq_run_btn.click(
    fn=lambda: gr.Row(visible=True),  # Show loading indicator
    outputs=dq_loading
    ).then(
    fn=lambda
        _conn_state, _dq_db, _dq_schema, _dq_table,
        _dq_selected_columns_null, _dq_null_threshold,
        _dq_check_table_overall_null_pct, _dq_table_null_threshold,
        _dq_value_range_table_input, _dq_date_range_table_input, _dq_selected_columns_regex,
        _dq_check_row_count, _dq_min_rows, _dq_check_duplicate_rows,
        _dq_check_value_range, _dq_check_date_range, _dq_check_regex_pattern, _dq_pattern,
        _dq_check_fk, _dq_fk_column, _dq_fk_ref_table, _dq_fk_ref_column,
        _dq_check_column_null_pct:
            DataQualityValidator(_conn_state).run_dq_checks(
                conn=_conn_state, database=_dq_db, schema=_dq_schema, table=_dq_table,
                dq_selected_columns_null=_dq_selected_columns_null, dq_null_threshold=_dq_null_threshold,
                dq_check_table_overall_null_pct=_dq_check_table_overall_null_pct, dq_table_null_threshold=_dq_table_null_threshold,
                dq_value_range_table=_dq_value_range_table_input, dq_date_range_table=_dq_date_range_table_input, dq_selected_columns_regex=_dq_selected_columns_regex,
                dq_check_row_count=_dq_check_row_count, dq_min_rows=_dq_min_rows, dq_check_duplicate_rows=_dq_check_duplicate_rows,
                dq_check_value_range=_dq_check_value_range, dq_check_date_range=_dq_check_date_range, dq_check_regex_pattern=_dq_check_regex_pattern, dq_pattern=_dq_pattern,
                dq_check_fk=_dq_check_fk, dq_fk_column=_dq_fk_column, dq_fk_ref_table=_dq_fk_ref_table, dq_fk_ref_column=_dq_fk_ref_column,
                dq_check_column_null_pct=_dq_check_column_null_pct
            ),
    inputs=[
        conn_state, dq_db, dq_schema, dq_table,
        dq_selected_columns_null, dq_null_threshold,
        dq_check_table_overall_null_pct, dq_table_null_threshold,
        dq_value_range_table,
        dq_date_range_table,
        dq_selected_columns_regex,
        dq_check_row_count, dq_min_rows,
        dq_check_duplicate_rows,
        dq_check_value_range,
        dq_check_date_range,
        dq_check_regex_pattern, dq_pattern,
        dq_check_fk, dq_fk_column, dq_fk_ref_table, dq_fk_ref_column,
        dq_check_column_null_pct
    ],
    outputs=[
        dq_summary,
        dq_details,
        dq_status,
        dq_score,
        dq_plot,
        dq_download_report_button,  # The download button visibility
        dq_report_filename_state,  # The state holding the filename
        dq_download_file  # The actual File component
    ]
    ).then(
    fn=lambda: gr.Row(visible=False),  # Hide loading indicator
    outputs=dq_loading
    )

    # Data Quality Download Handler
    dq_download_report_button.click(
        lambda filename: gr.File(value=filename, visible=True, label="Download Data Quality Report"),
        inputs=dq_report_filename_state,
        outputs=dq_download_file
    )


if __name__ == "__main__":
    try:
        from google.colab import output
        output.enable_custom_widget_manager()
        app.launch(server_name="0.0.0.0", server_port=7860)
    except:
        # Fallback for non-Colab environments
        app.launch(server_name="0.0.0.0", server_port=7860)
