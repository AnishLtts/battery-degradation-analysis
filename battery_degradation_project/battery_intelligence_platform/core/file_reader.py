from pathlib import Path

import pandas as pd


SUPPORTED_FILE_TYPES = {
    ".csv",
    ".xlsx",
    ".xls",
}


def load_battery_dataset(file_path):
    """
    Load a battery dataset from CSV or Excel.

    Parameters
    ----------
    file_path : str
        Full path to the battery dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded battery dataset.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset was not found: {file_path}"
        )

    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_FILE_TYPES:
        raise ValueError(
            f"Unsupported file type: {extension}\n"
            f"Currently supported: {sorted(SUPPORTED_FILE_TYPES)}"
        )

    if extension == ".csv":
        dataframe = read_csv_safely(file_path)

    elif extension == ".xlsx":
        dataframe = pd.read_excel(
            file_path,
            engine="openpyxl",
        )

    else:
        dataframe = pd.read_excel(
            file_path,
            engine="xlrd",
        )

    if dataframe.empty:
        raise ValueError(
            "The selected dataset is empty."
        )

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    return dataframe


def read_csv_safely(file_path):
    """
    Read a CSV using common encodings and separators.
    """

    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin-1",
    ]

    separators = [
        ",",
        ";",
        "\t",
    ]

    errors = []

    for encoding in encodings:
        for separator in separators:
            try:
                dataframe = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    sep=separator,
                    low_memory=False,
                )

                if dataframe.shape[1] > 1:
                    return dataframe

            except Exception as error:
                errors.append(str(error))

    try:
        return pd.read_csv(
            file_path,
            sep=None,
            engine="python",
            low_memory=False,
        )

    except Exception as error:
        errors.append(str(error))

        raise ValueError(
            "The CSV file could not be read. "
            "The file format, separator, or encoding may be unsupported."
        ) from error


def inspect_battery_dataset(dataframe):
    """
    Create a basic dataset inspection report.
    """

    numeric_columns = dataframe.select_dtypes(
        include="number"
    ).columns.tolist()

    non_numeric_columns = dataframe.select_dtypes(
        exclude="number"
    ).columns.tolist()

    missing_values = (
        dataframe.isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing_values = missing_values[
        missing_values > 0
    ].to_dict()

    completely_empty_columns = [
        column
        for column in dataframe.columns
        if dataframe[column].isna().all()
    ]

    constant_columns = [
        column
        for column in dataframe.columns
        if dataframe[column].nunique(
            dropna=True
        ) <= 1
    ]

    report = {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "column_names": dataframe.columns.tolist(),
        "numeric_columns": numeric_columns,
        "non_numeric_columns": non_numeric_columns,
        "missing_values": missing_values,
        "duplicate_rows": int(
            dataframe.duplicated().sum()
        ),
        "empty_columns": completely_empty_columns,
        "constant_columns": constant_columns,
        "memory_mb": round(
            dataframe.memory_usage(
                deep=True
            ).sum() / (1024 ** 2),
            2,
        ),
    }

    return report


def print_dataset_report(report):
    """
    Print the dataset inspection report.
    """

    print("\n" + "=" * 70)
    print("UNIVERSAL BATTERY DATASET INSPECTION")
    print("=" * 70)

    print(f"Rows                : {report['rows']}")
    print(f"Columns             : {report['columns']}")
    print(f"Duplicate rows      : {report['duplicate_rows']}")
    print(f"Memory usage        : {report['memory_mb']} MB")

    print("\nAVAILABLE COLUMNS")

    for column in report["column_names"]:
        print(f"  - {column}")

    print("\nNUMERIC COLUMNS")

    if report["numeric_columns"]:
        for column in report["numeric_columns"]:
            print(f"  - {column}")
    else:
        print("  No numeric columns detected.")

    print("\nTEXT OR CATEGORY COLUMNS")

    if report["non_numeric_columns"]:
        for column in report["non_numeric_columns"]:
            print(f"  - {column}")
    else:
        print("  No text or category columns detected.")

    print("\nMISSING VALUES")

    if report["missing_values"]:
        for column, count in report["missing_values"].items():
            print(f"  - {column}: {count}")
    else:
        print("  No missing values detected.")

    print("\nCOMPLETELY EMPTY COLUMNS")

    if report["empty_columns"]:
        for column in report["empty_columns"]:
            print(f"  - {column}")
    else:
        print("  No completely empty columns detected.")

    print("\nCONSTANT COLUMNS")

    if report["constant_columns"]:
        for column in report["constant_columns"]:
            print(f"  - {column}")
    else:
        print("  No constant columns detected.")

    print("=" * 70)


if __name__ == "__main__":
    dataset_path = input(
        "Paste the full path of a CSV or Excel battery dataset: "
    ).strip().strip('"')

    try:
        battery_data = load_battery_dataset(
            dataset_path
        )

        inspection_report = inspect_battery_dataset(
            battery_data
        )

        print_dataset_report(
            inspection_report
        )

        print("\nFIRST FIVE ROWS")
        print(battery_data.head())

    except Exception as error:
        print(f"\nDataset inspection failed: {error}")