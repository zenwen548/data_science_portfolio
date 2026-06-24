"""Seeded synthetic churn data generator for the portfolio dashboard."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


DEFAULT_ROW_COUNT = 2000
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = Path("projects/customer-churn/data")


def sigmoid(values):
    return 1 / (1 + np.exp(-values))


def choose(rng, values, size, probabilities=None):
    return rng.choice(values, size=size, p=probabilities)


def generate_churn_data(row_count=DEFAULT_ROW_COUNT, seed=DEFAULT_SEED):
    rng = np.random.default_rng(seed)
    customer_ids = [f"C{index:05d}" for index in range(1, row_count + 1)]
    subscription_ids = [f"S{index:05d}" for index in range(1, row_count + 1)]

    regions = choose(rng, ["North", "South", "East", "West"], row_count)
    state_map = {
        "North": ["IL", "MI", "MN", "WI"],
        "South": ["FL", "GA", "NC", "TX"],
        "East": ["MA", "NY", "PA", "VA"],
        "West": ["AZ", "CA", "OR", "WA"],
    }
    state_province = [
        rng.choice(state_map[region])
        for region in regions
    ]

    customer_segment = choose(
        rng,
        ["Small Business", "Mid-Market", "Enterprise"],
        row_count,
        [0.46, 0.34, 0.20],
    )
    contract_type = choose(
        rng,
        ["Month-to-month", "One year", "Two year"],
        row_count,
        [0.50, 0.30, 0.20],
    )
    payment_method = choose(
        rng,
        ["Electronic check", "Credit card", "Bank transfer", "Mailed check"],
        row_count,
        [0.34, 0.28, 0.25, 0.13],
    )
    internet_service = choose(
        rng,
        ["Fiber optic", "DSL", "None"],
        row_count,
        [0.48, 0.42, 0.10],
    )
    tech_support = choose(rng, ["Yes", "No"], row_count, [0.44, 0.56])
    paperless_billing = choose(rng, ["Yes", "No"], row_count, [0.64, 0.36])
    has_partner = choose(rng, ["Yes", "No"], row_count, [0.48, 0.52])
    has_dependents = choose(rng, ["Yes", "No"], row_count, [0.30, 0.70])
    senior_citizen = rng.binomial(1, 0.16, row_count)
    num_addon_services = rng.integers(0, 6, row_count)

    base_charge = rng.normal(68, 16, row_count)
    monthly_charges = (
        base_charge
        + np.where(internet_service == "Fiber optic", 18, 0)
        - np.where(internet_service == "None", 22, 0)
        + num_addon_services * 4.5
    ).clip(25, 150).round(2)

    risk_score = (
        -0.65
        + np.where(contract_type == "Month-to-month", 1.25, 0)
        + np.where(contract_type == "Two year", -0.95, 0)
        + np.where(payment_method == "Electronic check", 0.62, 0)
        + np.where(internet_service == "Fiber optic", 0.52, 0)
        + np.where(tech_support == "No", 0.68, -0.32)
        + np.where(paperless_billing == "Yes", 0.30, 0)
        + senior_citizen * 0.28
        - np.where(has_partner == "Yes", 0.26, 0)
        - np.where(has_dependents == "Yes", 0.25, 0)
        + num_addon_services * 0.08
        + (monthly_charges - monthly_charges.mean()) / monthly_charges.std() * 0.30
        + rng.normal(0, 0.62, row_count)
    )
    churn_probability = sigmoid(risk_score)
    churned = rng.binomial(1, churn_probability)

    tenure_days = rng.integers(60, 1800, row_count)
    start_dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(
        rng.integers(0, 1500, row_count), unit="D"
    )
    end_dates = [
        start + pd.Timedelta(days=int(days)) if churn else pd.NaT
        for start, days, churn in zip(start_dates, tenure_days, churned)
    ]

    num_support_calls = rng.poisson(
        1.2
        + np.where(tech_support == "No", 0.8, 0)
        + np.where(contract_type == "Month-to-month", 0.4, 0),
        row_count,
    ).clip(0, 12)
    avg_monthly_transactions = rng.normal(
        24
        + np.where(customer_segment == "Enterprise", 18, 0)
        + np.where(customer_segment == "Mid-Market", 7, 0)
        - churned * 2,
        8,
        row_count,
    ).clip(3, 85).round(0).astype(int)
    support_calls_per_day = (num_support_calls / tenure_days).round(4)
    avg_transactions_per_day = (avg_monthly_transactions / 30).round(4)

    was_referred = rng.binomial(1, 0.24, row_count)
    referral_code = [
        f"REF{rng.integers(100, 999)}" if referred else ""
        for referred in was_referred
    ]

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "customer_segment": customer_segment,
            "region": regions,
            "account_type": np.where(
                np.isin(contract_type, ["One year", "Two year"]), "Annual", "Monthly"
            ),
            "state_province": state_province,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "tech_support": tech_support,
            "num_addon_services": num_addon_services,
            "senior_citizen": senior_citizen,
            "has_partner": has_partner,
            "has_dependents": has_dependents,
            "paperless_billing": paperless_billing,
        }
    )
    subscriptions = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "subscription_id": subscription_ids,
            "subscription_start_date": start_dates.strftime("%Y-%m-%d"),
            "subscription_end_date": [
                "" if pd.isna(value) else value.strftime("%Y-%m-%d")
                for value in end_dates
            ],
            "subscription_cost": monthly_charges.round(0).astype(int),
            "monthly_charges": monthly_charges,
            "num_support_calls": num_support_calls,
            "support_calls_per_day": support_calls_per_day,
            "referral_code": referral_code,
            "avg_monthly_transactions": avg_monthly_transactions,
            "avg_transactions_per_day": avg_transactions_per_day,
            "tenure_days": tenure_days,
            "was_referred": was_referred,
            "churned": churned,
        }
    )
    return customers, subscriptions


def parse_args():
    parser = argparse.ArgumentParser(description="Generate seeded synthetic churn data.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROW_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    customers, subscriptions = generate_churn_data(args.rows, args.seed)
    customers.to_csv(output_dir / "sample_customers.csv", index=False)
    subscriptions.to_csv(output_dir / "sample_subscriptions.csv", index=False)
    print(
        f"Wrote {len(customers)} customers and {len(subscriptions)} subscriptions to {output_dir}"
    )


if __name__ == "__main__":
    main()
