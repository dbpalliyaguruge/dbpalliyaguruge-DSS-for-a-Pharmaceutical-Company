import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import io
from math import ceil, floor
import json

MONTH_OPTIONS = ["January", "February", "March", "April", "May", "June",
                 "July", "August", "September", "October", "November", "December"]
month_options = MONTH_OPTIONS

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


# ---------- Small helpers ----------
def normalize_product_types(df_products):
    """Convert Type column to lowercase and strip whitespace."""
    if "Type" in df_products.columns:
        df_products["Type"] = df_products["Type"].astype(
            str).str.lower().str.strip()
    return df_products


def prepare_display(df, first_col_name):
    """Reset index, rename first column properly (pandas 3 safe) and add a 'No.' column."""
    d = df.reset_index()
    d = d.rename(columns={d.columns[0]: first_col_name})
    d.insert(0, "No.", range(1, len(d) + 1))
    return d


def get_b_array(df_b):
    """Availability column -> clean float numpy array."""
    return pd.to_numeric(df_b["Availability"], errors="coerce").fillna(0).astype(float).values


def units_from_inventory(inv, prod_type, batch_const, size_per_unit):
    if size_per_unit == 0:
        return None
    base = inv * batch_const / size_per_unit
    if prod_type == "syrup":
        return base * 1000
    if prod_type == "tablet":
        return base * 1000000
    if prod_type in ("ointment", "cream"):
        return base * 1000
    return None


def material_status(values):
    out = []
    for rem in values:
        if rem > 1e-6:
            out.append("Sufficient")
        elif abs(rem) < 1e-6:
            out.append("DEPLETED")
        else:
            out.append("INSUFFICIENT")
    return out


def reset_simulation():
    """Reset the whole simulation from the current df_products / b."""
    s = st.session_state
    dfp = s.df_products
    s.current_month_idx = MONTH_OPTIONS.index(s.month)
    s.sim_month = 1
    s.sim_inventory = np.zeros(len(dfp))
    s.sim_raw_materials = np.array(s.b, dtype=float)
    s.sim_batch_const = dfp["BatchSizeConstant"].astype(float).to_numpy()
    s.sim_prod_size = dfp["ProdSizePerUnit"].astype(float).to_numpy()
    s.sim_product_type = np.array(
        dfp["Type"].astype(str).tolist(), dtype=object)
    s.sim_product_names = dfp.index.tolist()
    s.sim_month_inputs = {}
    s.sim_product_inputs = {}
    s.sim_plan_table = None
    s.planning_active = True
    s.show_continue_prompt = False
    s.planning_complete = False
    s.month_complete = False
    s.all_planning_complete = False
    s.current_month_start_inv = np.zeros(len(dfp))
    s.sim_monthly_history = []
    s.month_recorded = False


# ---------- Robust LP Solver ----------
def solve_lp_from_editors(A_edit, b_edit):
    try:
        materials_A = A_edit.iloc[:, 0].astype(str).values
        materials_b = b_edit.iloc[:, 0].astype(str).values

        common = set(materials_A) & set(materials_b)
        if not common:
            st.error("No common materials between A and b.")
            return np.zeros(A_edit.shape[1] - 1)

        idx_A = {mat: i for i, mat in enumerate(materials_A)}
        idx_b = {mat: i for i, mat in enumerate(materials_b)}

        common_list = sorted(common)

        A_rows = []
        b_vals = []
        for mat in common_list:
            i = idx_A[mat]
            j = idx_b[mat]
            row = pd.to_numeric(
                A_edit.iloc[i, 1:], errors='coerce').fillna(0).values
            A_rows.append(row)
            b_val = pd.to_numeric(b_edit.iloc[j, 1], errors='coerce')
            if pd.isna(b_val):
                b_val = 0.0
            b_vals.append(b_val)

        A = np.array(A_rows, dtype=float)
        b = np.array(b_vals, dtype=float)

        missing_in_b = set(materials_A) - common
        if missing_in_b:
            st.warning(f"Materials in A not in b: {missing_in_b} – ignored.")
        missing_in_A = set(materials_b) - common
        if missing_in_A:
            st.warning(f"Materials in b not in A: {missing_in_A} – ignored.")

        n_products = A.shape[1]
        if n_products == 0:
            st.error("No products found.")
            return np.array([])

        c = -np.ones(n_products)
        bounds = [(0, None)] * n_products

        options = {
            'primal_feasibility_tolerance': 1e-9,
            'dual_feasibility_tolerance': 1e-9,
            'optimality_tolerance': 1e-9
        }
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds,
                         method="highs", options=options)
        if result.success:
            return result.x
        else:
            st.error("Optimization failed: " + result.message)
            return np.zeros(n_products)
    except Exception as e:
        st.error(f"Solver error: {e}")
        return np.zeros(max(A_edit.shape[1] - 1, 0))


# ---------- Excel Loader ----------
def load_excel(uploaded_file):
    xl = pd.ExcelFile(uploaded_file)

    # ---------- Sheet "A" (Bill of Materials) ----------
    df_A_raw = pd.read_excel(xl, "A", header=None)

    # Product names are in row 1 (index 1) from column 1 onward
    product_names = df_A_raw.iloc[1, 1:].dropna().astype(str).tolist()

    # Data rows start at row 2 (index 2)
    data_rows = df_A_raw.iloc[2:].copy()
    data_rows = data_rows.dropna(subset=[0])
    raw_materials = data_rows[0].astype(str).tolist()
    data = data_rows.iloc[:, 1:1 + len(product_names)].astype(float).values

    df_A = pd.DataFrame(data, index=raw_materials, columns=product_names)
    df_A.index.name = "Raw Material"

    # ---------- Sheet "b" (Raw Material Availability) ----------
    df_b_raw = pd.read_excel(xl, "b", header=None)
    df_b_raw = df_b_raw.iloc[1:].dropna(subset=[0])
    avail_values = pd.to_numeric(
        df_b_raw.iloc[:, 1], errors="coerce").fillna(0).astype(float).values
    if df_b_raw.shape[1] >= 3:
        unit_values = df_b_raw.iloc[:, 2].fillna("").astype(str).values
    else:
        unit_values = [""] * len(avail_values)

    df_b = pd.DataFrame({
        "Availability": avail_values,
        "Unit": unit_values
    }, index=df_b_raw.iloc[:, 0].astype(str).values)
    df_b.index.name = "Raw Material"

    # ---------- Sheet "Products" ----------
    df_products = pd.read_excel(xl, "Products", index_col=0)
    df_products.index.name = "Product"
    if "Type" not in df_products.columns:
        df_products["Type"] = "unknown"
    if "BatchSizeConstant" not in df_products.columns:
        df_products["BatchSizeConstant"] = 1000.0
    if "ProdSizePerUnit" not in df_products.columns:
        df_products["ProdSizePerUnit"] = 1.0

    df_products = normalize_product_types(df_products)

    return df_A, df_b, df_products


# ---------- Excel Exporter ----------
def to_excel_bytes(df_A, df_b, df_products):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet "A": row 0 empty, row 1 product names, row 2+ data
        rows_A = []
        rows_A.append([])
        rows_A.append([""] + list(df_A.columns))
        for material in df_A.index:
            rows_A.append([material] + list(df_A.loc[material].values))
        pd.DataFrame(rows_A).to_excel(
            writer, sheet_name="A", index=False, header=False)

        # Sheet "b"
        rows_b = [["Material", "Availability", "Unit"]]
        for material in df_b.index:
            rows_b.append([
                material,
                df_b.loc[material, "Availability"],
                df_b.loc[material, "Unit"]
            ])
        pd.DataFrame(rows_b).to_excel(
            writer, sheet_name="b", index=False, header=False)

        # Sheet "Products"
        df_products.to_excel(writer, sheet_name="Products")
    return output.getvalue()


# ---------- Export current inventory to Excel ----------
def export_inventory_to_excel():
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        product_names = st.session_state.sim_product_names
        inventory = st.session_state.sim_inventory
        dfp = st.session_state.df_products

        product_types = []
        excess_units = []
        for k, prod in enumerate(product_names):
            prod_type = dfp.loc[prod, "Type"]
            product_types.append(prod_type)
            units = units_from_inventory(
                inventory[k], prod_type,
                dfp.loc[prod, "BatchSizeConstant"],
                dfp.loc[prod, "ProdSizePerUnit"]
            )
            excess_units.append(units if units is not None else "N/A")

        fg_df = pd.DataFrame({
            "Product": product_names,
            "Product Type": product_types,
            "Current Inventory (batches)": inventory,
            "Excess in Units": excess_units
        })
        fg_df.to_excel(writer, sheet_name="Finished Goods", index=False)

        raw_materials = st.session_state.df_A.index.tolist()
        initial_avail = st.session_state.b
        current_rem = st.session_state.sim_raw_materials
        unit_vals = st.session_state.df_b.loc[raw_materials, "Unit"].values

        raw_df = pd.DataFrame({
            "Material": raw_materials,
            "Unit": unit_vals,
            "Initial Availability": initial_avail,
            "Current Remaining": current_rem,
            "Status": material_status(current_rem)
        })
        raw_df.to_excel(writer, sheet_name="Raw Materials", index=False)

    return output.getvalue()


# ---------- Export monthly summaries to Excel ----------
def export_monthly_summaries_to_excel():
    """Export all recorded monthly summaries to an Excel file, one sheet per month."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for record in st.session_state.sim_monthly_history:
            month_name = record['month']
            df = record['table']
            sheet_name = f"{month_name} Summary"[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()


# ---------- Session Persistence Helpers ----------
def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def session_state_to_dict():
    """Convert relevant session state variables to a JSON-serializable dict."""
    def df_to_dict(df):
        if df is None:
            return None
        return df.reset_index().to_dict(orient='list')

    def arr(a):
        return a.tolist() if a is not None else None

    s = st.session_state
    state_dict = {
        "df_A": df_to_dict(s.df_A),
        "df_b": df_to_dict(s.df_b),
        "df_products": df_to_dict(s.df_products),
        "editor_A": df_to_dict(s.editor_A),
        "editor_b": df_to_dict(s.editor_b),
        "editor_products": df_to_dict(s.editor_products),
        "A": arr(s.A),
        "b": arr(s.b),
        "month": s.month,
        "current_month_idx": s.current_month_idx,
        "sim_inventory": arr(s.sim_inventory),
        "sim_raw_materials": arr(s.sim_raw_materials),
        "sim_batch_const": arr(s.sim_batch_const),
        "sim_prod_size": arr(s.sim_prod_size),
        "sim_product_type": arr(s.sim_product_type),
        "sim_product_names": s.sim_product_names,
        "sim_product_inputs": s.sim_product_inputs,
        "planning_active": s.planning_active,
        "show_continue_prompt": s.show_continue_prompt,
        "planning_complete": s.planning_complete,
        "month_complete": s.month_complete,
        "all_planning_complete": s.all_planning_complete,
        "edit_key": s.edit_key,
        "initial_optimal": arr(s.initial_optimal),
        "current_month_start_inv": arr(s.current_month_start_inv),
        "month_recorded": s.month_recorded,
    }

    serializable_history = []
    for record in s.sim_monthly_history:
        serializable_record = record.copy()
        if isinstance(serializable_record.get('table'), pd.DataFrame):
            serializable_record['table'] = serializable_record['table'].to_dict(
                orient='split')
        serializable_history.append(serializable_record)
    state_dict["sim_monthly_history"] = serializable_history

    return state_dict


def dict_to_session_state(data):
    """Restore session state from a dict (loaded from JSON)."""
    def rebuild_df(d):
        if d is None:
            return None
        df = pd.DataFrame.from_dict(d)
        if df.shape[1] >= 1:
            df = df.set_index(df.columns[0])
        return df

    def to_arr(v, dtype=None):
        return np.array(v, dtype=dtype) if v is not None else None

    s = st.session_state
    s.df_A = rebuild_df(data.get("df_A"))
    s.df_b = rebuild_df(data.get("df_b"))
    s.df_products = rebuild_df(data.get("df_products"))
    s.editor_A = rebuild_df(data.get("editor_A"))
    s.editor_b = rebuild_df(data.get("editor_b"))
    s.editor_products = rebuild_df(data.get("editor_products"))

    if s.df_products is not None:
        s.df_products = normalize_product_types(s.df_products)
    if s.editor_products is not None:
        s.editor_products = normalize_product_types(s.editor_products)

    s.A = to_arr(data.get("A"), float)
    s.b = to_arr(data.get("b"), float)
    loaded_month = data.get("month", "January")
    s.month = loaded_month
    s.pending_month = loaded_month
    s.current_month_idx = data.get("current_month_idx", 0)
    s.sim_inventory = to_arr(data.get("sim_inventory"), float)
    s.sim_raw_materials = to_arr(data.get("sim_raw_materials"), float)
    s.sim_batch_const = to_arr(data.get("sim_batch_const"), float)
    s.sim_prod_size = to_arr(data.get("sim_prod_size"), float)
    s.sim_product_type = to_arr(data.get("sim_product_type"), object)
    s.sim_product_names = data.get("sim_product_names", [])
    s.sim_product_inputs = data.get("sim_product_inputs", {})
    s.planning_active = data.get("planning_active", True)
    s.show_continue_prompt = data.get("show_continue_prompt", False)
    s.planning_complete = data.get("planning_complete", False)
    s.month_complete = data.get("month_complete", False)
    s.all_planning_complete = data.get("all_planning_complete", False)
    s.edit_key = data.get("edit_key", 0)
    s.initial_optimal = to_arr(data.get("initial_optimal"), float)
    s.current_month_start_inv = to_arr(
        data.get("current_month_start_inv"), float)
    s.month_recorded = data.get("month_recorded", False)

    rebuilt_history = []
    for record in data.get("sim_monthly_history", []):
        rebuilt_record = record.copy()
        table_dict = rebuilt_record.get('table')
        if isinstance(table_dict, dict) and 'data' in table_dict and 'columns' in table_dict:
            rebuilt_record['table'] = pd.DataFrame(
                data=table_dict['data'],
                columns=table_dict['columns'],
                index=table_dict.get('index')
            )
        rebuilt_history.append(rebuilt_record)
    s.sim_monthly_history = rebuilt_history


# ---------- Record Month ----------
def record_month(month_name, start_inv, end_inv, product_inputs, optimal_vec):
    """Create a summary table for the completed month and append to history."""
    products = st.session_state.sim_product_names
    prod_to_idx = {prod: i for i, prod in enumerate(products)}

    is_first_month = (len(st.session_state.sim_monthly_history) == 0)

    rows = []
    for prod_key, inp in product_inputs.items():
        prod = prod_key.replace("prod_", "", 1)
        if prod not in prod_to_idx:
            continue
        i = prod_to_idx[prod]

        demand = inp.get("demand", 0.0)
        planned = inp.get("total_production", 0.0)
        produced = inp.get("produced", False)

        status = "can produce" if produced else "can not be produced"

        if is_first_month:
            row = {
                "Product": prod,
                "Optimal batches": f"{optimal_vec[i]:.4f}" if optimal_vec is not None else "N/A",
                "Required batches": f"{demand:.4f}",
                "Adjusted": f"{planned:.4f}",
                "Status": status,
                "Inventory": f"{end_inv[i]:.4f}"
            }
        else:
            shortfall = max(0, demand - start_inv[i])
            row = {
                "Product": prod,
                "Brought forward Inv": f"{start_inv[i]:.4f}",
                "Required batches": f"{demand:.4f}",
                "Required after inventory": f"{shortfall:.4f}",
                "Adjusted": f"{planned:.4f}",
                "Status": status,
                "Final inventory": f"{end_inv[i]:.4f}"
            }
        rows.append(row)

    if is_first_month:
        desired_order = ["Product", "Optimal batches",
                         "Required batches", "Adjusted", "Status", "Inventory"]
    else:
        desired_order = ["Product", "Brought forward Inv", "Required batches",
                         "Required after inventory", "Adjusted", "Status", "Final inventory"]

    if rows:
        df = pd.DataFrame(rows)
        df = df[[col for col in desired_order if col in df.columns]]
    else:
        df = pd.DataFrame(columns=desired_order)

    st.session_state.sim_monthly_history.append({
        "month": month_name,
        "table": df,
        "is_first": is_first_month
    })


# ---------- Session State ----------
defaults = {
    "df_A": None,
    "df_b": None,
    "df_products": None,
    "editor_A": None,
    "editor_b": None,
    "editor_products": None,
    "A": None,
    "b": None,
    "edit_key": 0,
    "month": "January",
    "month_select": "January",
    "pending_month": None,
    "prev_month": None,
    "loaded_file_sig": None,
    "loaded_session_sig": None,
    "sim_month": 1,
    "sim_inventory": None,
    "sim_raw_materials": None,
    "sim_batch_const": None,
    "sim_prod_size": None,
    "sim_product_type": None,
    "sim_product_names": None,
    "sim_month_inputs": {},
    "sim_product_inputs": {},
    "sim_plan_table": None,
    "planning_active": True,
    "show_continue_prompt": False,
    "planning_complete": False,
    "current_month_idx": 0,
    "month_complete": False,
    "all_planning_complete": False,
    "initial_optimal": None,
    "current_month_start_inv": None,
    "sim_monthly_history": [],
    "month_recorded": False,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- Sidebar ----------
st.sidebar.title("Production Planning System")

# A loaded session may ask for a different month: apply it BEFORE the widget is created
if st.session_state.pending_month in MONTH_OPTIONS:
    st.session_state.month_select = st.session_state.pending_month
    st.session_state.prev_month = st.session_state.pending_month
    st.session_state.pending_month = None

selected_month = st.sidebar.selectbox(
    "Select Month", options=MONTH_OPTIONS, key="month_select")

# If the month changed, reset simulation to start from the new month
if st.session_state.prev_month is not None and selected_month != st.session_state.prev_month:
    st.session_state.month = selected_month
    if st.session_state.df_products is not None and st.session_state.b is not None:
        reset_simulation()
        st.warning(
            f"Start month changed to {selected_month}. All planning progress has been reset.")
st.session_state.month = selected_month
st.session_state.prev_month = selected_month

file = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])

# Load only when a NEW file is uploaded (otherwise every click would reset everything)
if file:
    file_sig = (file.name, file.size)
    if st.session_state.loaded_file_sig != file_sig:
        try:
            df_A, df_b, df_products = load_excel(file)
        except Exception as e:
            st.sidebar.error(f"Could not read the Excel file: {e}")
        else:
            st.session_state.df_A = df_A
            st.session_state.df_b = df_b
            st.session_state.df_products = df_products
            st.session_state.editor_A = df_A.copy()
            st.session_state.editor_b = df_b.copy()
            st.session_state.editor_products = df_products.copy()
            st.session_state.A = df_A.values
            st.session_state.b = get_b_array(df_b)
            st.session_state.edit_key = 0
            reset_simulation()
            st.session_state.initial_optimal = solve_lp_from_editors(
                st.session_state.editor_A.reset_index(),
                st.session_state.editor_b.reset_index()
            )
            st.session_state.loaded_file_sig = file_sig

if st.session_state.df_A is not None:
    excel_bytes = to_excel_bytes(
        st.session_state.df_A,
        st.session_state.df_b,
        st.session_state.df_products
    )
    st.sidebar.download_button(
        label="📥 Download Current Data as Excel",
        data=excel_bytes,
        file_name=f"production_data_{st.session_state.month}.xlsx",
        mime=XLSX_MIME
    )

# ---------- Session Persistence UI ----------
with st.sidebar.expander("💾 Save / Load Session", expanded=False):
    if st.session_state.df_A is not None:
        session_json = json.dumps(
            session_state_to_dict(), indent=2, default=_json_default)
        st.download_button(
            label="💾 Download session backup",
            data=session_json,
            file_name="session_backup.json",
            mime="application/json",
            key="download_session"
        )
    else:
        st.caption("Upload an Excel file first to enable saving.")

    uploaded_session = st.file_uploader(
        "📂 Load Session", type=["json"], key="session_uploader")
    if uploaded_session is not None:
        session_sig = (uploaded_session.name, uploaded_session.size)
        if st.session_state.loaded_session_sig != session_sig:
            try:
                data = json.loads(uploaded_session.getvalue().decode("utf-8"))
                dict_to_session_state(data)
                st.session_state.loaded_session_sig = session_sig
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load session: {e}")
        else:
            st.success("Session loaded successfully!")

# ---------- Editing Section ----------
if st.session_state.df_A is not None:
    with st.sidebar.expander("Edit Data", expanded=True):
        # ----- Bill of Materials (BOM) Matrix -----
        st.subheader("Bill of Materials (BOM) Matrix")
        df_A_display = prepare_display(
            st.session_state.editor_A, "Raw Material")
        A_edit = st.data_editor(
            df_A_display,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            key=f"A_editor_{st.session_state.edit_key}"
        )
        if "No." in A_edit.columns:
            A_edit = A_edit.drop(columns=["No."])
        st.session_state.editor_A = A_edit.set_index(A_edit.columns[0])

        # ----- Add Product -----
        col_new_prod, col_new_prod_name = st.columns([1, 3])
        with col_new_prod:
            if st.button("➕ Add Product"):
                new_prod_name = st.session_state.get(
                    "new_product_name", "").strip()
                if not new_prod_name:
                    st.warning("Enter a product name.")
                elif new_prod_name in st.session_state.editor_A.columns:
                    st.warning("Product already exists.")
                else:
                    st.session_state.editor_A[new_prod_name] = 0.0
                    new_prod_row = pd.DataFrame({
                        "Type": ["unknown"],
                        "BatchSizeConstant": [1000.0],
                        "ProdSizePerUnit": [1.0]
                    }, index=pd.Index([new_prod_name], name="Product"))
                    st.session_state.editor_products = pd.concat(
                        [st.session_state.editor_products, new_prod_row]
                    )
                    st.session_state.edit_key += 1
                    st.rerun()
        with col_new_prod_name:
            st.text_input("New product name", key="new_product_name",
                          placeholder="e.g., Paracetamol")

        # ----- Remove Product -----
        if len(st.session_state.editor_products) > 0:
            st.subheader("Remove Product")
            prod_to_remove = st.selectbox(
                "Select product to remove",
                options=st.session_state.editor_products.index.tolist(),
                key="remove_product_select"
            )
            if st.button("❌ Remove Product"):
                if prod_to_remove in st.session_state.editor_A.columns:
                    st.session_state.editor_A = st.session_state.editor_A.drop(columns=[
                                                                               prod_to_remove])
                    st.session_state.editor_products = st.session_state.editor_products.drop(
                        index=prod_to_remove)
                    st.session_state.edit_key += 1
                    st.rerun()
                else:
                    st.error(
                        f"Product '{prod_to_remove}' not found in matrix A.")
        else:
            st.info("No products to remove.")

        st.markdown("---")

        # ----- Raw material Availability -----
        st.subheader("Raw material Availability")
        df_b_display = prepare_display(
            st.session_state.editor_b, "Raw Material")
        b_edit = st.data_editor(
            df_b_display,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            key=f"b_editor_{st.session_state.edit_key}"
        )
        if "No." in b_edit.columns:
            b_edit = b_edit.drop(columns=["No."])
        st.session_state.editor_b = b_edit.set_index(b_edit.columns[0])

        # ----- Product Details -----
        st.subheader("Product Details")
        prod_column_config = {
            "Type": st.column_config.TextColumn("Type", required=True),
            "BatchSizeConstant": st.column_config.NumberColumn("Batch Constant", min_value=0.0),
            "ProdSizePerUnit": st.column_config.NumberColumn("Size per Unit", min_value=0.0)
        }
        prod_display = prepare_display(
            st.session_state.editor_products, "Product")
        prod_edit = st.data_editor(
            prod_display,
            num_rows="dynamic",
            width="stretch",
            column_config=prod_column_config,
            hide_index=True,
            key=f"prod_editor_{st.session_state.edit_key}"
        )
        if "No." in prod_edit.columns:
            prod_edit = prod_edit.drop(columns=["No."])
        st.session_state.editor_products = prod_edit.set_index(
            prod_edit.columns[0])

        if st.button("Save Changes"):
            st.session_state.editor_products = normalize_product_types(
                st.session_state.editor_products)

            st.session_state.df_A = st.session_state.editor_A.copy()
            st.session_state.df_b = st.session_state.editor_b.copy()
            st.session_state.df_products = st.session_state.editor_products.copy()
            st.session_state.A = st.session_state.df_A.values
            st.session_state.b = get_b_array(st.session_state.df_b)

            reset_simulation()
            st.session_state.initial_optimal = solve_lp_from_editors(
                st.session_state.editor_A.reset_index(),
                st.session_state.editor_b.reset_index()
            )

            st.success(
                "Changes saved – simulation has been reset to start fresh with the new data.")

        st.download_button(
            label="📥 Download Updated Data as Excel",
            data=to_excel_bytes(
                st.session_state.df_A,
                st.session_state.df_b,
                st.session_state.df_products
            ),
            file_name=f"production_data_{st.session_state.month}_updated.xlsx",
            mime=XLSX_MIME,
            key="download_after_save"
        )

# ---------- Main App ----------
st.title("Pharmaceutical Production Planning")

if st.session_state.df_products is None:
    st.info("Upload Excel file first")
    st.stop()

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Initial Optimization", "Monthly Planning", "Inventory Details", "Monthly Summaries"])

# ---------- Tab 1: Initial Optimization ----------
with tab1:
    st.header("Initial Optimization")
    st.caption(f"Planning for: **{st.session_state.month}**")

    if st.session_state.sim_raw_materials is not None:
        materials = st.session_state.df_A.index.tolist()
        b_current_df = pd.DataFrame({
            'Raw Material': materials,
            'Availability': st.session_state.sim_raw_materials
        })
        x = solve_lp_from_editors(
            st.session_state.editor_A.reset_index(),
            b_current_df
        )
    else:
        x = solve_lp_from_editors(
            st.session_state.editor_A.reset_index(),
            st.session_state.editor_b.reset_index()
        )

    if len(x) == 0:
        st.warning("No products – check your data.")
    else:
        df_lp = pd.DataFrame({
            "Product": st.session_state.editor_products.index,
            "Optimal Batches": [f"{val:.6f}" for val in x]
        })
        df_lp.insert(0, "No.", range(1, len(df_lp) + 1))
        st.dataframe(df_lp, width="stretch", hide_index=True)

# ---------- Tab 2: Monthly Planning ----------
with tab2:
    st.header("Production Simulation")
    current_month_name = MONTH_OPTIONS[st.session_state.current_month_idx]
    st.caption(f"Planning for: **{current_month_name}**")

    # Ensure simulation state is consistent
    if (st.session_state.sim_product_names is None
            or len(st.session_state.sim_product_names) != len(st.session_state.df_products)):
        reset_simulation()

    x_opt = solve_lp_from_editors(
        st.session_state.editor_A.reset_index(),
        st.session_state.editor_b.reset_index()
    )

    # ------------------------------------------------------------------
    # MULTI-MONTH PLANNING STATE MACHINE
    # ------------------------------------------------------------------
    if st.session_state.all_planning_complete:
        st.success("✅ All months planning completed.")
        if st.session_state.sim_monthly_history:
            for record in st.session_state.sim_monthly_history:
                with st.expander(f"📅 {record['month']} Summary"):
                    st.dataframe(record['table'],
                                 width="stretch", hide_index=True)
        else:
            st.info("No monthly data recorded.")
        if st.button("🔄 Start New Planning"):
            reset_simulation()
            st.rerun()

    elif st.session_state.month_complete:
        st.subheader(f"End of {current_month_name} – Inventory Status")

        fg_df = pd.DataFrame({
            "Product": st.session_state.sim_product_names,
            "Current Inventory (batches)": [f"{val:.4f}" for val in st.session_state.sim_inventory]
        })
        fg_df.insert(0, "No.", range(1, len(fg_df) + 1))
        st.dataframe(fg_df, width="stretch", hide_index=True)

        if st.session_state.df_A is not None:
            unit_vals = st.session_state.df_b.loc[st.session_state.df_A.index, "Unit"].values
            raw_df = pd.DataFrame({
                "Idx": range(1, len(st.session_state.df_A.index) + 1),
                "Material": st.session_state.df_A.index,
                "Unit": unit_vals,
                "Initial Availability": [f"{val:.4f}" for val in st.session_state.b],
                "Current Remaining": [f"{val:.4f}" for val in st.session_state.sim_raw_materials],
                "Status": material_status(st.session_state.sim_raw_materials)
            })
            st.dataframe(raw_df, width="stretch", hide_index=True)

        if not st.session_state.month_recorded:
            record_month(
                month_name=current_month_name,
                start_inv=st.session_state.current_month_start_inv,
                end_inv=st.session_state.sim_inventory,
                product_inputs=st.session_state.sim_product_inputs,
                optimal_vec=st.session_state.initial_optimal
            )
            st.session_state.month_recorded = True

        st.write("---")
        st.write("**Do you want to proceed to the next month?**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, go to next month"):
                if st.session_state.current_month_idx + 1 < len(MONTH_OPTIONS):
                    st.session_state.current_month_idx += 1
                    st.session_state.month_complete = False
                    st.session_state.planning_active = True
                    st.session_state.show_continue_prompt = False
                    st.session_state.current_month_start_inv = st.session_state.sim_inventory.copy()
                    st.session_state.sim_product_inputs = {}
                    st.session_state.month_recorded = False
                    st.rerun()
                else:
                    st.warning("No more months. Planning finished.")
                    st.session_state.all_planning_complete = True
                    st.rerun()
        with col2:
            if st.button("❌ No, finish planning"):
                st.session_state.all_planning_complete = True
                st.rerun()

    else:
        if st.session_state.show_continue_prompt:
            st.info("Do you want to plan another product for this month?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes"):
                    st.session_state.show_continue_prompt = False
                    st.rerun()
            with col2:
                if st.button("❌ No, finish this month"):
                    st.session_state.show_continue_prompt = False
                    st.session_state.month_complete = True
                    st.rerun()
        else:
            product_names = st.session_state.sim_product_names
            selected_product = st.selectbox(
                "Select Product", product_names, key="sim_selected_product")
            i = product_names.index(selected_product)

            prod_key = f"prod_{selected_product}"
            if prod_key not in st.session_state.sim_product_inputs:
                st.session_state.sim_product_inputs[prod_key] = {
                    "method": 1,
                    "order_qty": 0,
                    "direct_demand": 0.0,
                    "rounding_choice": "Ceil",
                    "production_qty": 0.0,
                    "demand": 0.0,
                    "total_production": 0.0,
                    "produced": False,
                }
            inp = st.session_state.sim_product_inputs[prod_key]

            current_inv = st.session_state.sim_inventory[i]
            st.info(f"**Current inventory:** {current_inv:.4f} batches")

            method = st.radio(
                "Order input method",
                options=[1, 2],
                format_func={1: "Batch Calculation", 2: "Direct Entry"}.get,
                index=inp["method"] - 1,
                key=f"method_{selected_product}"
            )
            inp["method"] = method

            if method == 1:
                prod_type = st.session_state.sim_product_type[i]
                batch_const = st.session_state.sim_batch_const[i]
                prod_size_per_unit = st.session_state.sim_prod_size[i]

                st.info(f"**Product Type:** {prod_type}")
                st.info(f"**Batch Constant:** {batch_const}")
                st.info(f"**Size per Unit:** {prod_size_per_unit}")

                order_qty = st.number_input(
                    "Monthly order quantity (units)",
                    value=inp["order_qty"],
                    step=1,
                    key=f"order_qty_{selected_product}"
                )
                inp["order_qty"] = order_qty

                if prod_type == "syrup":
                    demand = (prod_size_per_unit * order_qty) / \
                        (batch_const * 1000)
                elif prod_type == "tablet":
                    demand = (prod_size_per_unit * order_qty) / \
                        (batch_const * 1000000)
                elif prod_type in ("ointment", "cream"):
                    demand = (prod_size_per_unit * order_qty) / \
                        (batch_const * 1000)
                else:
                    demand = 0.0
                    st.warning(
                        f"Unknown product type '{prod_type}' – demand set to 0.")
            else:
                direct_demand = st.number_input(
                    "Enter demand (required batches)",
                    value=float(inp["direct_demand"]),
                    key=f"direct_demand_{selected_product}"
                )
                inp["direct_demand"] = direct_demand
                demand = direct_demand

            demand = float(demand)
            inp["demand"] = demand
            st.write(f"**Required number of batches (demand):** {demand:.4f}")

            shortfall = max(0.0, demand - current_inv)
            st.write(f"**Shortfall (demand - inventory):** {shortfall:.4f}")

            rounding_options = ["Ceil", "Floor", "Ceil+1",
                                "Floor-1", "Enter value directly"]
            default_index = (rounding_options.index(inp["rounding_choice"])
                             if inp["rounding_choice"] in rounding_options else 0)
            rounding_choice = st.selectbox(
                "Rounding / adjustment for production quantity",
                options=rounding_options,
                index=default_index,
                key=f"rounding_{selected_product}"
            )
            inp["rounding_choice"] = rounding_choice

            if rounding_choice == "Ceil":
                production_qty = ceil(shortfall)
            elif rounding_choice == "Floor":
                production_qty = floor(shortfall)
            elif rounding_choice == "Ceil+1":
                production_qty = ceil(shortfall) + 1
            elif rounding_choice == "Floor-1":
                production_qty = max(0, floor(shortfall) - 1)
            else:
                production_qty = st.number_input(
                    "Enter production quantity",
                    value=float(inp.get("production_qty", shortfall)),
                    key=f"prod_qty_{selected_product}"
                )
            inp["production_qty"] = production_qty

            st.write(f"**Planned production batches:** {production_qty:.4f}")

            if len(st.session_state.sim_monthly_history) == 0:
                if len(x_opt) > i:
                    opt_val = x_opt[i]
                    st.write(
                        f"**Optimal batches (from initial optimization):** {opt_val:.6f}")
                    if production_qty <= opt_val + 1e-9:
                        st.success("✅ Production is within the optimal plan.")
                    else:
                        st.error(
                            "❌ Production exceeds optimal plan; raw materials may be insufficient.")

            if st.session_state.df_A is not None and i < len(st.session_state.df_A.columns):
                A_col = st.session_state.df_A.iloc[:, i].values.astype(float)
                required_raw = A_col * production_qty
                current_raw = st.session_state.sim_raw_materials
                sufficient = bool(np.all(required_raw <= current_raw + 1e-9))

                if sufficient:
                    st.success(
                        "✅ Currently available raw materials are sufficient for this production.")
                else:
                    st.error(
                        "❌ Currently available raw materials are INSUFFICIENT for this production.")
                    insufficient_list = []
                    for idx, mat in enumerate(st.session_state.df_A.index):
                        if required_raw[idx] > current_raw[idx] + 1e-9:
                            insufficient_list.append({
                                "Material": mat,
                                "Required": required_raw[idx],
                                "Available": current_raw[idx],
                                "Shortfall": required_raw[idx] - current_raw[idx]
                            })
                    if insufficient_list:
                        st.warning("Insufficient raw materials:")
                        ins_df = pd.DataFrame(insufficient_list)
                        ins_df.insert(0, "No.", range(1, len(ins_df) + 1))
                        st.dataframe(ins_df, width="stretch", hide_index=True)

                new_inv = current_inv + production_qty - demand
                st.write(f"**Projected new inventory:** {new_inv:.4f} batches")

                if sufficient:
                    if st.button("✅ Apply Production", key="apply_production"):
                        st.session_state.sim_raw_materials = st.session_state.sim_raw_materials - required_raw
                        st.session_state.sim_inventory[i] = new_inv
                        inp["produced"] = True
                        inp["total_production"] = inp.get(
                            "total_production", 0.0) + float(production_qty)
                        st.session_state.show_continue_prompt = True
                        st.rerun()
                else:
                    if st.button("⏭️ Skip Product", key="skip_product"):
                        st.session_state.show_continue_prompt = True
                        st.rerun()
            else:
                st.warning("Product data not available.")

        if st.button("🔄 Reset Simulation to Initial State"):
            reset_simulation()
            st.rerun()

# ---------- Tab 3: Inventory Details ----------
with tab3:
    st.header("Inventory Details")

    st.subheader("Finished Goods Inventory")
    product_names = st.session_state.sim_product_names
    inventory = st.session_state.sim_inventory
    dfp = st.session_state.df_products

    product_types = []
    excess_units = []
    for k, prod in enumerate(product_names):
        prod_type = dfp.loc[prod, "Type"]
        product_types.append(prod_type)
        units = units_from_inventory(
            inventory[k], prod_type,
            dfp.loc[prod, "BatchSizeConstant"],
            dfp.loc[prod, "ProdSizePerUnit"]
        )
        excess_units.append(f"{units:.2f}" if units is not None else "N/A")

    fg_df = pd.DataFrame({
        "Product": product_names,
        "Product Type": product_types,
        "Current Inventory (batches)": [f"{val:.4f}" for val in inventory],
        "Excess in Units": excess_units
    })
    fg_df.insert(0, "No.", range(1, len(fg_df) + 1))
    st.dataframe(fg_df, width="stretch", hide_index=True)

    if st.session_state.df_A is not None:
        st.subheader("Raw Material Inventory")
        unit_vals = st.session_state.df_b.loc[st.session_state.df_A.index, "Unit"].values
        raw_df = pd.DataFrame({
            "Idx": range(1, len(st.session_state.df_A.index) + 1),
            "Material": st.session_state.df_A.index,
            "Unit": unit_vals,
            "Initial Availability": [f"{val:.4f}" for val in st.session_state.b],
            "Current Remaining": [f"{val:.4f}" for val in st.session_state.sim_raw_materials],
            "Status": material_status(st.session_state.sim_raw_materials)
        })
        st.dataframe(raw_df, width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("Export Inventory Report")
    st.download_button(
        label="📥 Download Inventory Report as Excel",
        data=export_inventory_to_excel(),
        file_name=f"inventory_report_{st.session_state.month}.xlsx",
        mime=XLSX_MIME
    )

# ---------- Tab 4: Monthly Summaries ----------
with tab4:
    st.header("Monthly Production Summaries")

    if not st.session_state.sim_monthly_history:
        st.info(
            "No monthly data recorded yet. Complete a month in the Monthly Planning tab to see summaries.")
    else:
        for record in st.session_state.sim_monthly_history:
            month_name = record['month']
            df_table = record['table']

            with st.expander(f"📅 {month_name} Summary", expanded=False):
                if df_table.empty:
                    st.caption("No products were planned this month.")
                else:
                    st.dataframe(df_table, width="stretch", hide_index=True)

        st.markdown("---")
        st.download_button(
            label="📥 Download All Monthly Summaries as Excel",
            data=export_monthly_summaries_to_excel(),
            file_name="monthly_summaries.xlsx",
            mime=XLSX_MIME
        )
