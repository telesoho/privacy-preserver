
# from pyspark.sql.functions import PandasUDFType, lit, pandas_udf
from .mondrian_utils.utility import *


def k_anonymizer(df, k, feature_columns, sensitive_column, categorical):

    if sensitive_column not in df.columns:
        raise AnonymizeError("No Such Sensitive Column")

    for fcolumn in feature_columns:
        if fcolumn not in df.columns:
            raise AnonymizeError("No Such Feature Column :"+fcolumn)

    full_spans = get_full_span(df, categorical)
    partitions = partition_dataset(
        df, k, None, None,  categorical, feature_columns, sensitive_column, full_spans)

    return anonymizer(df, partitions, feature_columns, sensitive_column, categorical)


def l_diversity_anonymizer(df, k, l, feature_columns, sensitive_column, categorical):

    if sensitive_column not in df.columns:
        raise AnonymizeError("No Such Sensitive Column")

    for fcolumn in feature_columns:
        if fcolumn not in df.columns:
            raise AnonymizeError("No Such Feature Column :"+fcolumn)

    full_spans = get_full_span(df, categorical)
    partitions = partition_dataset(
        df, k, l, None,  categorical, feature_columns, sensitive_column, full_spans)

    return anonymizer(df, partitions, feature_columns, sensitive_column, categorical)


def t_closeness_anonymizer(df, k, t, feature_columns, sensitive_column, categorical):

    if sensitive_column not in df.columns:
        raise AnonymizeError("No Such Sensitive Column")

    for fcolumn in feature_columns:
        if fcolumn not in df.columns:
            raise AnonymizeError("No Such Feature Column :"+fcolumn)

    full_spans = get_full_span(df, categorical)
    partitions = partition_dataset(
        df, k, None, t,  categorical, feature_columns, sensitive_column, full_spans)

    return anonymizer(df, partitions, feature_columns, sensitive_column, categorical)


def k_anonymizer_w_user(df, k, feature_columns, sensitive_column, categorical):

    if sensitive_column not in df.columns:
        raise AnonymizeError("No Such Sensitive Column")

    for fcolumn in feature_columns:
        if fcolumn not in df.columns:
            raise AnonymizeError("No Such Feature Column :"+fcolumn)

    full_spans = get_full_span(df, categorical)
    partitions = partition_dataset(
        df, k, None, None,  categorical, feature_columns, sensitive_column, full_spans)

    return anonymize_w_user(
        df, partitions, feature_columns, sensitive_column, categorical)


def l_diversity_anonymizer_w_user(df, k, l, feature_columns, sensitive_column, categorical):

    if sensitive_column not in df.columns:
        raise AnonymizeError("No Such Sensitive Column")

    for fcolumn in feature_columns:
        if fcolumn not in df.columns:
            raise AnonymizeError("No Such Feature Column :"+fcolumn)

    full_spans = get_full_span(df, categorical)
    partitions = partition_dataset(
        df, k, l, None,  categorical, feature_columns, sensitive_column, full_spans)

    return anonymize_w_user(
        df, partitions, feature_columns, sensitive_column, categorical)


def t_closeness_anonymizer_w_user(df, k, t, feature_columns, sensitive_column, categorical):

    if sensitive_column not in df.columns:
        raise AnonymizeError("No Such Sensitive Column")

    for fcolumn in feature_columns:
        if fcolumn not in df.columns:
            raise AnonymizeError("No Such Feature Column :"+fcolumn)

    full_spans = get_full_span(df, categorical)
    partitions = partition_dataset(
        df, k, None, t,  categorical, feature_columns, sensitive_column, full_spans)

    return anonymize_w_user(
        df, partitions, feature_columns, sensitive_column, categorical)


class Preserver:

    @staticmethod
    def k_anonymize(df, k, feature_columns, sensitive_column, categorical, schema):

        # def anonymize(pdf):
        a_df = k_anonymizer(df, k, feature_columns,
                            sensitive_column, categorical)
        return a_df

    @staticmethod
    def k_anonymize_w_user(df, k, feature_columns, sensitive_column, categorical, schema):
        a_df = k_anonymizer_w_user(df, k, feature_columns,
                                    sensitive_column, categorical)
        return a_df

    @staticmethod
    def l_diversity(df, k, l, feature_columns, sensitive_column, categorical, schema):
        a_df = l_diversity_anonymizer(df, k, l, feature_columns,
                                        sensitive_column, categorical)
        return a_df

    @staticmethod
    def l_diversity_w_user(df, k, l, feature_columns, sensitive_column, categorical, schema):
        a_df = l_diversity_anonymizer_w_user(df, k, l, feature_columns,
                                                sensitive_column, categorical)
        return a_df

    @staticmethod
    def t_closeness(df, k, t, feature_columns, sensitive_column, categorical, schema):
        a_df = t_closeness_anonymizer(df, k, t, feature_columns,
                                        sensitive_column, categorical)
        return a_df

    @staticmethod
    def t_closeness_w_user(df, k, t, feature_columns, sensitive_column, categorical, schema):
        a_df = t_closeness_anonymizer_w_user(df, k, t, feature_columns,
                                                sensitive_column, categorical)
        return a_df


    @staticmethod
    def anonymize_user(df, k, user, usercolumn_name, sensitive_column, categorical, schema, random=False):
        a_df = user_anonymizer(
            df, k, user, usercolumn_name, sensitive_column, categorical, random)
        return a_df
