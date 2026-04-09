import math
import pandas as pd
import numpy as np

# ERC20 Decimals mapping identical to Rust get_token_decimals()
TOKEN_DECIMALS = {
    "0xdac17f958d2ee523a2206206994597c13d831ec7": 6,
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": 6,
    "0x6c3ea9036406852006290770bedfcaba0e23a0e8": 6,
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": 18,
    "0xb8c77482e45f1f44de1745f52c74426c631bdd52": 18,
    "0x056fd409e1d7a124bd7017459dfea2f387b6d5cd": 2,
    "0xdb25f211ab05b1c97d595516f45794528a807ad8": 2,
    "0x6b175474e89094c44da98b954eedeac495271d0f": 18,
    "0x4c9edd5852cd905f086c759e8383e09bff1e68b3": 18,
    "0x853d955acef822db058eb8505911ed77f175b99e": 18,
    "0xae7ab96520de3a18e5e111b5eaab095312d7fe84": 18,
    "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0": 18,
    "0xae78736cd615f374d3085123a210448e74fc6393": 18,
    "0xbe9895146f7af43049ca1c1ae358b0541ea49704": 18,
    "0xb50721bcf8d664c30412cfbc6cf7a15145234ad1": 18,
    "0x3c3a81e81dc49a522a592e7622a7e711c06bf354": 18,
    "0xf57e7e7c23978c3caec3c3548e3d615c346e79ff": 18,
    "0x514910771af9ca656af840dff83e8264ecf986ca": 18,
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984": 18,
    "0x7d1afa7b718fb893db30a3abc0cfc608aacfebb0": 18,
    "0x455e53cbb86018ac2b8092fdcd39d8444affc3f6": 18,
    "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9": 18,
    "0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2": 18,
    "0x5a98fcbea516cf06857215779fd812ca3bef1b32": 18,
    "0xc011a73ee8576fb46f5e1c5751ca3b9fe0af2a6f": 18,
    "0x95ad61b0a150d79219dcf64e1e6cc01f0b64c4ce": 18,
    "0x6982508145454ce325ddbe47a25d4ec3d2311933": 18,
    "0xcf0c122c6b73ff809c693db761e7baebe62b6a2e": 9,
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": 8,
    "0xa0b73e1ff0b80914ab6fe0444e65848c4c34450b": 8,
}


def extract_features(
    df: pd.DataFrame, target_address: str, sequence_id: int
) -> pd.DataFrame:
    df = df.sort_values(by=["block_timestamp", "transaction_index"]).reset_index(
        drop=True
    )
    print(f"{target_address = }")
    target_address = target_address.lower()

    # --- Internal State Variables ---
    first_seen = 0
    last_seen_generic = 0
    db_nonce = 0
    fail_count = 0
    in_count = 0
    out_count = 0
    sum_sq_diff = 0
    sum_diff = 0
    contract_count = 0
    last_send_ts = 0
    session_tx_count = 0
    lifetime_erc20 = 0
    prev_gap = 0

    SESSION_GAP_SECONDS = 900

    # Expected output columns
    result_keys = [
        "correspondent_address",
        "final_counterparty",
        "related_contract",
        "hour_utc",
        "day_of_week",
        "day_of_month",
        "month",
        "tx_cost_eth",
        "is_sender",
        "is_successful",
        "input_len",
        "method_id",
        "is_contract_call",
        "tx_type",
        "is_self_tx",
        "is_early_block",
        "s_age_days",
        "s_time_since_last",
        "s_time_since_send",
        "s_nonce",
        "s_fail_rate",
        "s_io_ratio",
        "s_freq_std_dev",
        "s_contract_count",
        "s_session_depth",
        "s_is_new_session",
        "s_inter_arrival_ratio",
        "s_lifetime_erc20",
        "token_value_log",
        "input_entropy",
        "s_total_interactions_log",
    ]
    results = {key: [] for key in result_keys}

    def extract_state_features(is_sender_role: bool, input_nonce: int, ts: int):
        """Calculates features from current state BEFORE applying DB updates."""
        is_new = first_seen == 0
        time_since_send = (
            max(0.0, float(ts - last_send_ts))
            if not is_new and last_send_ts > 0
            else 0.0
        )
        time_since_generic = (
            max(0.0, float(ts - last_seen_generic)) if not is_new else 0.0
        )

        is_new_session = is_sender_role and (
            is_new or time_since_send > SESSION_GAP_SECONDS
        )

        interval_count = max(0, out_count - 1)
        freq_std_dev = 0.0
        if interval_count > 0:
            mean = sum_diff / interval_count
            mean_sq = sum_sq_diff / interval_count
            var = max(0.0, mean_sq - mean**2)
            freq_std_dev = math.sqrt(var)

        current_session_depth = 0
        if is_sender_role:
            current_session_depth = 1 if is_new_session else session_tx_count + 1

        current_gap_raw = ts - last_seen_generic if last_seen_generic > 0 else 0
        acceleration = (current_gap_raw / prev_gap) if prev_gap > 0 else 1.0

        total_ops = in_count + out_count
        io_ratio = (in_count / total_ops) if total_ops > 0 else 0.5
        current_nonce = max(db_nonce, input_nonce) if is_sender_role else db_nonce

        return {
            "s_age_days": 0.0 if is_new else (ts - first_seen) / 86400.0,
            "s_time_since_last": time_since_generic,
            "s_time_since_send": time_since_send,
            "s_nonce": current_nonce,
            "s_fail_rate": (fail_count / out_count) if out_count > 0 else 0.0,
            "s_io_ratio": io_ratio,
            "s_freq_std_dev": freq_std_dev,
            "s_contract_count": contract_count,
            "s_session_depth": current_session_depth,
            "s_is_new_session": is_new_session,
            "s_inter_arrival_ratio": acceleration,
            "s_lifetime_erc20": lifetime_erc20,
            "s_total_interactions_log": math.log(total_ops + 1.0),
        }

    def apply_state_update(
        is_sender_role: bool,
        tx_failed: bool,
        is_contract_creation: bool,
        is_erc_activity: bool,
        input_nonce: int,
        ts: int,
    ):
        """Mutates global state exactly like RocksDB put()."""
        nonlocal \
            first_seen, \
            last_seen_generic, \
            db_nonce, \
            fail_count, \
            in_count, \
            out_count
        nonlocal sum_sq_diff, sum_diff, contract_count, last_send_ts, session_tx_count
        nonlocal lifetime_erc20, prev_gap

        is_new = first_seen == 0
        new_first = ts if is_new else first_seen
        new_last_seen_generic = ts

        new_nonce = db_nonce
        new_fail = fail_count
        new_in = in_count
        new_out = out_count
        new_sum_sq = sum_sq_diff
        new_sum_diff = sum_diff
        new_contract_count = contract_count
        new_last_send_ts = last_send_ts
        new_sess_depth = session_tx_count
        new_erc20 = lifetime_erc20
        new_prev_gap = prev_gap

        current_gap_raw = ts - last_seen_generic if last_seen_generic > 0 else 0
        if last_seen_generic > 0:
            new_prev_gap = current_gap_raw

        if is_sender_role:
            new_nonce = input_nonce if input_nonce > 0 else new_nonce + 1
            new_out += 1
            if tx_failed:
                new_fail += 1
            if is_contract_creation:
                new_contract_count += 1
            if is_erc_activity:
                new_erc20 += 1

            if last_send_ts > 0:
                diff = ts - last_send_ts if ts > last_send_ts else 0
                new_sum_diff += diff
                new_sum_sq += diff**2

            new_last_send_ts = ts

            time_since_send_pre = (
                (ts - last_send_ts) if not is_new and last_send_ts > 0 else 0.0
            )
            is_new_sess = is_new or time_since_send_pre > SESSION_GAP_SECONDS
            if is_new_sess:
                new_sess_depth = 1
            else:
                new_sess_depth += 1
        else:
            new_in += 1

        first_seen = new_first
        last_seen_generic = new_last_seen_generic
        db_nonce = new_nonce
        fail_count = new_fail
        in_count = new_in
        out_count = new_out
        sum_sq_diff = new_sum_sq
        sum_diff = new_sum_diff
        contract_count = new_contract_count
        last_send_ts = new_last_send_ts
        session_tx_count = new_sess_depth
        lifetime_erc20 = new_erc20
        prev_gap = new_prev_gap

    # --- Processing Loop ---
    for i, row in df.iterrows():
        ts = int(row["block_timestamp"].timestamp())
        tx_index = int(row["transaction_index"])
        input_nonce = int(row.get("nonce", 0))

        from_addr = str(row["from_address"]).lower()
        to_raw = str(row.get("to_address", "")).lower()
        if to_raw in ("nan", "none"):
            to_raw = ""

        # Determine strict failure status matching Pre-Byzantium fallback
        status = row.get("receipt_status")
        gas_limit = int(row.get("gas", 0))
        gas_used = int(row.get("receipt_gas_used", 0))

        if pd.isna(status) or status is None or str(status).lower() == "nan":
            tx_failed = gas_limit > 0 and gas_used >= gas_limit
        else:
            tx_failed = float(status) == 0.0
        is_successful = not tx_failed

        # Safe Input Bytes Parsing
        input_hex = str(row.get("input", "0x")).lower()
        if input_hex in ("nan", "none"):
            input_hex = "0x"
        if input_hex.startswith("0x"):
            input_hex = input_hex[2:]
        try:
            input_bytes = bytes.fromhex(input_hex)
        except ValueError:
            input_bytes = b""

        input_len = len(input_bytes)

        # Determine Method & Tx Types
        method_id = 0
        method_str = ""
        if input_len >= 4:
            method_str = input_bytes[:4].hex()
            try:
                method_id = int(method_str, 16)
            except ValueError:
                pass

        is_contract_creation = len(to_raw) < 5
        ca_raw = str(row.get("receipt_contract_address", "")).lower()
        ca = ca_raw if ca_raw not in ("nan", "none", "") else ""

        is_transfer = method_str == "a9059cbb"
        is_transfer_from = method_str == "23b872dd"
        is_approval = method_str == "095ea7b3"
        is_erc_activity = is_transfer or is_transfer_from or is_approval

        # Byte-level ERC20 Extraction
        is_erc20_transfer = False
        effective_to = to_raw
        token_val_log = 0.0

        if input_len >= 68 and is_transfer:
            effective_to = "0x" + input_bytes[16:36].hex()
            is_erc20_transfer = True
            val_u128 = int.from_bytes(input_bytes[52:68], byteorder="big")
            decimals = TOKEN_DECIMALS.get(to_raw, 18)
            token_val_log = math.log((val_u128 / (10**decimals)) + 1.0)

        elif input_len >= 100 and is_transfer_from:
            effective_to = "0x" + input_bytes[48:68].hex()
            is_erc20_transfer = True
            val_u128 = int.from_bytes(input_bytes[84:100], byteorder="big")
            decimals = TOKEN_DECIMALS.get(to_raw, 18)
            token_val_log = math.log((val_u128 / (10**decimals)) + 1.0)

        # Entropy Calculation
        entropy = 0.0
        if input_len >= 4:
            data = input_bytes[4:]
            if data:
                counts = {}
                for b in data:
                    counts[b] = counts.get(b, 0) + 1
                length = float(len(data))
                for count in counts.values():
                    p = count / length
                    entropy -= p * math.log2(p)

        # Output Strings Format matching Rust
        final_counterparty = effective_to if is_erc20_transfer else "None"
        is_self_tx = not is_contract_creation and (from_addr == to_raw)

        if is_contract_creation:
            related_contract = ca if ca else "None"
            correspondent_address = "None"
        else:
            if is_erc_activity or (input_len > 0 and to_raw):
                related_contract = to_raw
            else:
                related_contract = "None"
            correspondent_address = to_raw if to_raw else "None"

        # --- Roles & State Evaluation ---
        is_target_sender = from_addr == target_address
        receiver_addr = ca if is_contract_creation else effective_to
        is_target_receiver = receiver_addr == target_address

        # Generate features exactly ONCE per transaction from Target's perspective
        feat = extract_state_features(
            is_sender_role=is_target_sender, input_nonce=input_nonce, ts=ts
        )
        is_nonce_replay = (
            is_target_sender and (first_seen > 0) and (input_nonce <= db_nonce)
        )

        dt = pd.to_datetime(ts, unit="s", utc=True)
        tx_cost_eth = (gas_used * int(row.get("gas_price", 0))) / 1e18

        # --- Populate Row Data ---
        results["correspondent_address"].append(correspondent_address)
        results["final_counterparty"].append(final_counterparty)
        results["related_contract"].append(related_contract)
        results["hour_utc"].append(dt.hour)
        results["day_of_week"].append(dt.dayofweek + 1)
        results["day_of_month"].append(dt.day)
        results["month"].append(dt.month)
        results["tx_cost_eth"].append(tx_cost_eth)
        results["is_sender"].append(is_target_sender)
        results["is_successful"].append(is_successful)
        results["input_len"].append(input_len)
        results["method_id"].append(method_id)
        results["is_contract_call"].append(input_len >= 4)
        results["tx_type"].append(int(row.get("transaction_type", 0)))
        results["is_self_tx"].append(is_self_tx)
        results["is_early_block"].append(tx_index <= 2)
        results["token_value_log"].append(token_val_log)
        results["input_entropy"].append(entropy)

        for k, v in feat.items():
            results[k].append(v)

        # --- Apply Database Mutations Sequence ---
        if is_target_sender:
            if not is_nonce_replay:
                apply_state_update(
                    True,
                    tx_failed,
                    is_contract_creation,
                    is_erc_activity,
                    input_nonce,
                    ts,
                )

            # Double write logic for self-transactions
            if not tx_failed and receiver_addr != "" and is_target_receiver:
                apply_state_update(
                    False,
                    tx_failed,
                    is_contract_creation,
                    is_erc_activity,
                    input_nonce,
                    ts,
                )
        else:
            if not tx_failed and is_target_receiver:
                apply_state_update(
                    False,
                    tx_failed,
                    is_contract_creation,
                    is_erc_activity,
                    input_nonce,
                    ts,
                )

    for key, values in results.items():
        df[key] = values

    df["sequenceId"] = np.repeat(sequence_id, df.shape[0])
    df["itemPosition"] = np.arange(df.shape[0])

    string_cols = [
        "correspondent_address",
        "final_counterparty",
        "related_contract",
        "method_id",
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None").astype(str)

    # 2. Cast booleans to int (Sequifier/Polars handles Int64 better than Bool for mapping)
    bool_cols = [
        "is_sender",
        "is_successful",
        "is_contract_call",
        "is_self_tx",
        "is_early_block",
        "s_is_new_session",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # 3. Existing fix for timestamp
    df["block_timestamp"] = df["block_timestamp"].astype("int64") // 10**6

    return df[
        ["sequenceId", "itemPosition"]
        + [c for c in df.columns if c not in ["sequenceId", "itemPosition"]]
    ]
