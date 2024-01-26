import itertools
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from Function.Project_VNg63984_Function import select_variance, select_best


def preprocess_data_for_sales_forecasting():
    df = pd.read_csv("Dataset/DataCoSupplyChainDataset.csv",
                     encoding="ISO-8859-1")
    df.columns = df.columns.str.lower().str.replace(
        " ", "_").str.replace(r"\(|\)", "").str.strip()

    # removing null value
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='all', inplace=True)

    # drop zipcode and name columns
    df.drop(['order_zipcode', 'customer_zipcode', 'customer_fname',
            'customer_lname'], axis=1, inplace=True)

    # drop duplicates
    df.drop_duplicates(keep='last', inplace=True)

    # drop uneccessary columns
    columns_to_drop = [
        "category_id", "customer_id", "customer_email", "customer_password", "department_id", "order_customer_id",
        "order_id", "order_item_cardprod_id", "order_item_id", "product_card_id", "product_category_id",
        "product_image", "product_status", "product_name", "customer_street",]

    df.drop(columns=columns_to_drop, inplace=True)

    # pre-processing for sales forecasting linear regression

    # copy the df into the new dataframe for further processing
    df_cleaned = df.copy()

    # compute pairwise correlation of columns
    corr = df_cleaned.corr()

    # create a list for columns to be dropped due to high correlation
    drop_columns = []

    # identifying highly correlated columns
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i, j] >= 0.9:
                colname = corr.columns[j]
                if colname not in drop_columns and colname != 'sales':
                    drop_columns.append(colname)

    # dropping highly correlated columns
    df_cleaned.drop(drop_columns, axis=1, inplace=True)

    # eliminate outlier from the target
    Q1 = df_cleaned.sales.quantile(0.25)
    Q3 = df_cleaned.sales.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # remove outliers
    df_cleaned = df_cleaned[(df_cleaned.sales >= lower_bound) & (
        df_cleaned.sales <= upper_bound)]

    # create a list of categorical columns
    cat_columns = df_cleaned.select_dtypes(include=['object']).columns

    # change all the values in categorical columns to replace hypen ‘-‘ into underscore and space to underscore
    df_cleaned[cat_columns] = df_cleaned[cat_columns]\
        .apply(lambda x: x.str.strip().str.replace("-", "_").str.replace(" ", "_").str.lower())

    # dropping 'customer_city', 'order_city', and 'order_state'
    df_cleaned = df_cleaned.drop(
        ['customer_city', 'order_city', 'order_state', 'order_country'], axis=1)

    # divide the customer_state into customer_region
    northeast = ['me', 'nh', 'vt', 'ma', 'ct', 'ri', 'ny', 'nj', 'pa']
    midwest = ['oh', 'mi', 'in', 'wi', 'il',
               'mn', 'ia', 'mo', 'nd', 'sd', 'ne', 'ks']
    south = ['de', 'md', 'dc', 'va', 'wv', 'nc', 'sc', 'ga',
             'fl', 'ky', 'tn', 'al', 'ms', 'ar', 'la', 'tx', 'ok']
    west = ['mt', 'id', 'wy', 'co', 'nm', 'az',
            'ut', 'nv', 'ca', 'or', 'wa', 'ak', 'hi']

    df_cleaned['customer_region'] = ['northeast' if state in northeast else 'midwest' if state in midwest
                                     else 'south' if state in south else 'west'
                                     if state in west else 'territories' if state == 'pr'
                                     else 'other' for state in df_cleaned.customer_state]

    df_cleaned.drop('customer_state', axis=1, inplace=True)

    df_cleaned.drop('category_name', axis=1, inplace=True)

    # reduce complexity of department_name column:

    department_mapping = {
        'fan_shop': 'sports_&_outdoor',
        'golf': 'sports_&_outdoor',
        'outdoors': 'sports_&_outdoor',
        'fitness': 'sports_&_outdoor',
        'apparel': 'apparel',
        'footwear': 'footwear',
        'discs_shop': 'entertainment',
        'book_shop': 'entertainment',
        'technology': 'technology',
        'pet_shop': 'personal_care_&_lifestyle',
        'health_and_beauty': 'personal_care_&_lifestyle'
    }

    df_cleaned['department_name'] = df_cleaned['department_name'].map(
        department_mapping)

    # reduce complexity of order_region
    order_region_mapping = {
        'central_america': 'latin_america',
        'south_america': 'latin_america',
        'caribbean': 'latin_america',

        'west_of_usa': 'north_america',
        'east_of_usa': 'north_america',
        'us_center': 'north_america',
        'south_of_usa': 'north_america',
        'canada': 'north_america',

        'western_europe': 'europe',
        'northern_europe': 'europe',
        'southern_europe': 'europe',
        'eastern_europe': 'europe',

        'southeast_asia': 'east_asia',
        'eastern_asia': 'east_asia',

        'south_asia': 'south_asia',

        'west_asia': 'west_asia',
        'central_asia': 'west_asia',

        'west_africa': 'africa',
        'north_africa': 'africa',
        'east_africa': 'africa',
        'central_africa': 'africa',
        'southern_africa': 'africa',

        'oceania': 'oceania'
    }

    df_cleaned.order_region = df_cleaned['order_region'].map(
        order_region_mapping)

    # replacing 'EE. UU.' with 'united_states'
    df_cleaned.customer_country = df_cleaned.customer_country.replace(
        'ee._uu.', 'united_states')

    # replace underscores with spaces for the date columns
    df_cleaned['order_date_dateorders'] = df_cleaned['order_date_dateorders'].str.replace(
        '_', ' ')
    df_cleaned['shipping_date_dateorders'] = df_cleaned['shipping_date_dateorders'].str.replace(
        '_', ' ')

    # convert to datetime
    df_cleaned['order_date_dateorders'] = pd.to_datetime(
        df_cleaned['order_date_dateorders'])
    df_cleaned['shipping_date_dateorders'] = pd.to_datetime(
        df_cleaned['shipping_date_dateorders'])

    # for order_date_dateorders
    df_cleaned['order_year'] = df_cleaned['order_date_dateorders'].dt.year
    df_cleaned['order_month'] = df_cleaned['order_date_dateorders'].dt.month
    df_cleaned['order_day'] = df_cleaned['order_date_dateorders'].dt.day
    df_cleaned['order_dayofweek'] = df_cleaned['order_date_dateorders'].dt.dayofweek
    df_cleaned['order_is_weekend'] = df_cleaned['order_dayofweek'].isin([
                                                                        5, 6]).astype(int)

    # for shipping_date_dateorders
    df_cleaned['shipping_year'] = df_cleaned['shipping_date_dateorders'].dt.year
    df_cleaned['shipping_month'] = df_cleaned['shipping_date_dateorders'].dt.month
    df_cleaned['shipping_day'] = df_cleaned['shipping_date_dateorders'].dt.day
    df_cleaned['shipping_dayofweek'] = df_cleaned['shipping_date_dateorders'].dt.dayofweek
    df_cleaned['shipping_is_weekend'] = df_cleaned['shipping_dayofweek'].isin([
                                                                              5, 6]).astype(int)

    df_cleaned['days_to_ship'] = (
        df_cleaned['shipping_date_dateorders'] - df_cleaned['order_date_dateorders']).dt.days

    # drop the original dates columns
    df_cleaned.drop(
        ['order_date_dateorders', 'shipping_date_dateorders'], axis=1, inplace=True)

    # drop the latitude and longitude columns
    df_cleaned.drop(['latitude', 'longitude'], axis=1, inplace=True)

    # create the dummies from the dataframe
    df_transformed = pd.get_dummies(df_cleaned, drop_first=True)

    columns_to_log = ['benefit_per_order', 'order_item_discount', 'order_item_profit_ratio',
                      'sales', 'order_item_quantity', 'days_for_shipment_scheduled']

    for col in columns_to_log:
        df_transformed[col] = df_transformed[col].apply(lambda x: np.log1p(x))

    # drop null values
    df_transformed.dropna(axis=0, how='any', inplace=True)

    # drop sales_per_customer to avoid multicollinearity
    df_transformed.drop('sales_per_customer', axis=1, inplace=True)

    # set target
    target = 'sales'

    # replace +inf and -inf with NaN
    df_transformed.replace([np.inf, -np.inf], np.nan, inplace=True)

    df_transformed.dropna(inplace=True)

    # scale the data first
    scaler = StandardScaler()
    df_transformed_scaled = pd.DataFrame(scaler.fit_transform(
        df_transformed), columns=df_transformed.columns)

    # variance threshold selection
    threshold_variance = 0.1
    df_variance = select_variance(
        df_transformed_scaled, target, threshold_variance)

    # K-Best selection

    X = df_transformed_scaled.drop(target, axis=1)
    y = df_transformed_scaled[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    selected_features_kbest = select_best(X_train, y_train, 12)
    df_selKBest = df_transformed_scaled[selected_features_kbest]
    return df_transformed_scaled, X, df_variance, df_selKBest


def preprocess_data_for_delay_detection():

    df = pd.read_csv("Dataset/DataCoSupplyChainDataset.csv",
                     encoding="ISO-8859-1")
    df.columns = df.columns.str.lower().str.replace(
        " ", "_").str.replace(r"\(|\)", "").str.strip()

    # removing null value
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='all', inplace=True)

    # drop zipcode and name columns
    df.drop(['order_zipcode', 'customer_zipcode', 'customer_fname',
            'customer_lname'], axis=1, inplace=True)

    # drop duplicates
    df.drop_duplicates(keep='last', inplace=True)

    # drop uneccessary columns
    columns_to_drop = [
        "category_id", "customer_id", "customer_email", "customer_password", "department_id", "order_customer_id",
        "order_id", "order_item_cardprod_id", "order_item_id", "product_card_id", "product_category_id",
        "product_image", "product_status", "product_name", "customer_street",]

    df.drop(columns=columns_to_drop, inplace=True)

    target_delay = 'late_delivery_risk'
    features = df.drop(target_delay, axis=1)

    # drop 'delivery_status' as it is directly related to the target
    features = df.drop('delivery_status', axis=1)

    # label encoding for categorical variables
    labelencoder = LabelEncoder()
    categorical_cols = features.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        features[col] = labelencoder.fit_transform(features[col])

    # feature selection based on importance
    F_values, p_values = f_regression(features, df[target_delay])

    # creating a df of features with their corresponding F-statistic and P-values for feature selection
    f_reg_results = [(i, v, z) for i, v, z in itertools.zip_longest(
        features.columns, F_values,  ['%.3f' % p for p in p_values])]
    f_reg_results = pd.DataFrame(f_reg_results, columns=[
                                 'Variable', 'F_Value', 'P_Value'])

    f_reg_results = f_reg_results.sort_values(by=['P_Value'])
    f_reg_results.P_Value = f_reg_results.P_Value.astype(float)
    f_reg_results = f_reg_results[f_reg_results.P_Value < 0.06]
    f_reg_list = f_reg_results.Variable.values

    df_delaydetection = features[f_reg_list]

    return df, df_delaydetection


# global variables
df_transformed, df_original, df_variance, df_selKBest = preprocess_data_for_sales_forecasting()
df, df_delaydetection = preprocess_data_for_delay_detection()
df1 = pd.read_csv("Dataset/DataCoSupplyChainDataset.csv",
                  encoding="ISO-8859-1")
