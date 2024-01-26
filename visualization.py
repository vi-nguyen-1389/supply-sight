import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from preprocessing import df


def generate_heatmap():

    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(),
                cbar=True,
                annot=True,
                square=True,
                fmt='.1g',
                linewidths=0.5,
                annot_kws={'size': 8},
                cmap="Blues"
                )
    plt.savefig("static/img/corr_heatmap.png")


def create_confusion_matrix_heatmap(cm, classifier_name):
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, annot_kws={
                "size": 12}, cbar=False, square=True, fmt="d", cmap="Blues")
    plt.title(f'Confusion Matrix for {classifier_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save with a transparent background
    plt.savefig(f'static/img/cm_{classifier_name}.png', transparent=True)

    plt.close()


def generate_bar_chart(df, x, y, title, file_name, fig_width, fig_height, hue=None, color=None):
    plt.figure(figsize=(fig_width, fig_height))
    sns.barplot(x=x, y=y, hue=hue, data=df, palette=color or "viridis")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'static/img/{file_name}.png')
    plt.close()


def generate_pie_chart(df, values, names, title, file_name, fig_width, fig_height):
    plt.figure(figsize=(fig_width, fig_height))
    plt.pie(df[values], labels=df[names], autopct='%1.1f%%',
            startangle=140, colors=sns.color_palette("Set2"))
    plt.title(title)
    plt.savefig(f'static/img/{file_name}.png')
    plt.close()


def generate_horizontal_bar_chart(df, x, y, title, file_name, fig_width, fig_height, hue=None, color=None):
    plt.figure(figsize=(fig_width, fig_height))
    sns.barplot(x=x, y=y, hue=hue, data=df,
                palette=color or "viridis", orient='h')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'static/img/{file_name}.png')
    plt.close()


def create_visualizations(df):

    # Delivery Status Bar Plot
    data_delivery_status = df.groupby(['Delivery Status'])['Order Id'].count().reset_index(
        name='Number of Orders').sort_values(by='Number of Orders', ascending=False)
    generate_bar_chart(data_delivery_status, 'Delivery Status', 'Number of Orders',
                       'Delivery Status Distribution', 'delivery_status_distribution', 10, 6)

    # Customer Segment Pie Chart
    data_customer_segment = df.groupby(['Customer Segment'])[
        'Order Id'].count().reset_index(name='Number of Orders')
    generate_pie_chart(data_customer_segment, 'Number of Orders',
                       'Customer Segment', 'Customer Segment Distribution', 'customer_segment', 8, 6)

    # Category Distribution - Top 10 Categories
    top_categories = df.groupby('Category Name')['Order Id'].count().nlargest(
        10).reset_index(name='Number of Orders')
    generate_bar_chart(top_categories, 'Category Name', 'Number of Orders',
                       'Top 10 Categories by Number of Orders', 'top_categories', 12, 6)

    # Delivery Status by Region Bar Chart
    data_delivery_status_region = df.groupby(['Delivery Status', 'Order Region'])['Order Id'].count(
    ).reset_index(name='Number of Orders').sort_values(by='Number of Orders', ascending=False)
    generate_bar_chart(data_delivery_status_region, 'Delivery Status', 'Number of Orders',
                       'Delivery Status by Region', 'delivery_status_region', 12, 7, hue='Order Region', color='plasma')

    # Top 20 Countries by Number of Orders
    country_name_mapping = {
        'Estados Unidos': 'United States',
        'Francia': 'France',
        'México': 'Mexico',
        'Alemania': 'Germany',
        'Australia': 'Australia',
        'Brasil': 'Brazil',
        'Reino Unido': 'United Kingdom',
        'China': 'China',
        'Italia': 'Italy',
        'India': 'India',
        'Indonesia': 'Indonesia',
        'España': 'Spain',
        'El Salvador': 'El Salvador',
        'República Dominicana': 'Dominican Republic',
        'Honduras': 'Honduras',
        'Cuba': 'Cuba',
        'Turquía': 'Turkey',
        'Nicaragua': 'Nicaragua',
        'Guatemala': 'Guatemala',
        'Nigeria': 'Nigeria'
    }

    # Apply the mapping
    df['Order Country'] = df['Order Country'].replace(country_name_mapping)

    data_countries = df.groupby(['Order Country'])['Order Id'].count().reset_index(
        name='Number of Orders').sort_values(by='Number of Orders', ascending=False)
    generate_bar_chart(data_countries.head(20), 'Order Country', 'Number of Orders',
                    'Top 20 Countries by Number of Orders', 'top_countries_orders', 10, 6, hue=None, color='Blues')

                    
    # Sales by Product Name
    data_product_sales = df.groupby(['Product Name'])['Sales'].sum().reset_index(
        name='Sales of Orders').sort_values(by='Sales of Orders', ascending=False)
    generate_horizontal_bar_chart(data_product_sales.head(
        10), 'Sales of Orders', 'Product Name', 'Top 10 Products by Sales', 'top_product_sales', 12, 7, color='magma')

    # Sales by Order Region
    data_region_sales = df.groupby(['Order Region'])['Sales'].sum().reset_index(
        name='Sales of Orders').sort_values(by='Sales of Orders', ascending=False)
    generate_bar_chart(data_region_sales, 'Order Region', 'Sales of Orders',
                       'Sales by Order Region', 'region_sales', 12, 7, color='Blues')

    # Sales Over Time (Yearly)
    df['order_date'] = pd.to_datetime(df['order date (DateOrders)'])
    df['Year'] = df['order_date'].dt.year
    df_yearly_sales = df.groupby(
        'Year')['Sales'].sum().reset_index(name='Sales of Orders')
    generate_bar_chart(df_yearly_sales, 'Year', 'Sales of Orders',
                       'Yearly Sales Overview', 'yearly_sales', 12, 8)  # Increased figure size

    # Average Order Value by Category - Top 10 Categories
    df['Order Count'] = df.groupby('Category Name')[
        'Order Id'].transform('count')
    data_avg_order_value = df.groupby('Category Name').agg(
        {'Sales': 'sum', 'Order Count': 'first'})
    data_avg_order_value['Avg Order Value'] = data_avg_order_value['Sales'] / \
        data_avg_order_value['Order Count']
    top_avg_order_value = data_avg_order_value.nlargest(
        10, 'Avg Order Value').reset_index()
    generate_bar_chart(top_avg_order_value, 'Category Name', 'Avg Order Value',
                       'Top 10 Categories by Avg Order Value', 'top_avg_order_value', 12, 6, color='coolwarm')

    # Monthly Sales Trend
    df['Month'] = df['order_date'].dt.month
    df_monthly_sales = df.groupby('Month')['Sales'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    plt.plot(df_monthly_sales['Month'],
             df_monthly_sales['Sales'], marker='o', color='teal')
    plt.title('Monthly Sales Trend')
    plt.xlabel('Month')
    plt.ylabel('Total Sales')
    plt.xticks(range(1, 13))
    plt.grid(True)
    plt.savefig('static/img/monthly_sales_trend.png')
    plt.close()
