import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import webbrowser
import os
import datetime as dt
import pytz
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

apps_df = pd.read_csv('Data/Play Store Data.csv')
reviews_df = pd.read_csv('Data/User Reviews.csv')


# Removing entries having no Rating values
apps_df = apps_df.dropna(subset=['Rating'])

# filling missing values with most frequent value of each column
for column in apps_df.columns :
    apps_df[column].fillna(apps_df[column].mode()[0],inplace=True)

# Dropping duplicate rows
apps_df.drop_duplicates(inplace=True)
# Removing Rating values greater than 5
apps_df=apps_df[apps_df['Rating']<=5]

# Dropping entries having no Translated Reviews 
reviews_df.dropna(subset=['Translated_Review'],inplace=True)

#Convert the Installs columns to numeric by removing commas and +
apps_df['Installs']=apps_df['Installs'].str.replace(',','').str.replace('+','').astype(int)

#Convert Price column to numeric after removing $
apps_df['Price']=apps_df['Price'].str.replace('$','').astype(float)

apps_df['Reviews'] = apps_df['Reviews'].astype(int)

#Revenue column
apps_df['Revenue']=apps_df['Price']*apps_df['Installs']

# Convert Last updated Column to a date format
apps_df['Last Updated'] = apps_df['Last Updated'].astype(str)
apps_df['Last Updated']=pd.to_datetime(apps_df['Last Updated'], format= '%B %d, %Y')

def rating_group(rating):
    if rating <= 2:
        return '1-2 stars'
    elif rating <= 3:
        return '2-3 stars'
    elif rating <= 4:
        return '3-4 stars'
    elif rating <= 5:
        return '4-5 stars'
    
apps_df['Rating_group']=apps_df['Rating'].apply(rating_group)

#Convert Size column
def convert_size(size):
    if 'M' in size:
        return float(size.replace('M',''))
    elif 'k' in size:
        return float(size.replace('k',''))/1024
    else:
        return np.nan
apps_df['Size']=apps_df['Size'].apply(convert_size)

#Revenue column
apps_df['Revenue']=apps_df['Price']*apps_df['Installs']

# Filling Null Values with mode
for column in apps_df.columns :
    apps_df[column].fillna(apps_df[column].mode()[0],inplace=True)

# Merging the datasets into one
merged_df=pd.merge(apps_df,reviews_df,on='App',how='inner')

html_files_path="./"
if not os.path.exists(html_files_path):
    os.makedirs(html_files_path)
plot_containers=""

# Save each Plotly figure to an HTML file
def save_plot_as_html(fig, filename, insight):
    global plot_containers
    filepath = os.path.join(html_files_path, filename)
    html_content = pio.to_html(fig, full_html=False, include_plotlyjs='inline')
    # Append the plot and its insight to plot_containers
    plot_containers += f"""
    <div class="plot-container" id="{filename}" onclick="openPlot('{filename}')">
        <div class="plot">{html_content}</div>
        <div class="insights">{insight}</div>
    </div>
    """
    fig.write_html(filepath, full_html=False, include_plotlyjs='inline')


# Get the current time in UTC
utc_time = dt.datetime.now(pytz.utc)

# Convert UTC time to Indian Standard Time (IST)
current_time = utc_time.astimezone(pytz.timezone('Asia/Kolkata')).time()
print(current_time)


#Figure 1 Coding
filter1a  = merged_df[merged_df['Reviews'] > 1000]
filter1a = filter1a[["Category","Rating_group","Sentiment"]]
# filter1a.info()
top_categories = filter1a['Category'].value_counts().nlargest(5).index
filter1a = filter1a[filter1a['Category'].isin(top_categories)]

sentiment_dist = (filter1a.groupby(['Category', 'Rating_group', 'Sentiment']).size().reset_index(name='Count'))

#Figure 1

fig1=px.bar(
    sentiment_dist,
    x="Rating_group",
    y="Count",
    labels={'Count': 'Sentiment Count', 'Rating_group': 'Rating Group'},
    color='Sentiment',
    title='Sentiment Distribution by Rating Groups of Top 5 Categories',
    color_discrete_map={'Positive': 'green', 'Neutral': 'blue', 'Negative': 'red'},
    facet_col="Category",
    width=1000,
    height=400,
    barmode="stack",
    
)
# Modify facet column annotations to remove "Category = " and avoid overlapping
fig1.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1].strip()))

fig1.for_each_xaxis(lambda x: x.update(title=None)),
fig1.add_annotation(
    x=0.5,
    y=-0.30,
    showarrow=False,
    text="Rating Group",
    xref="paper",
    yref="paper",
    font=dict(size=14)
)
fig1.update_xaxes(tickangle=35)


# Save plot as a html file
save_plot_as_html(fig1,"Task 1.html","Sentiment Distribution by Rating Groups of Top 5 Categories")

#Figure 2 Coding
filter_df2 = apps_df[~apps_df['Category'].str.startswith(('A', 'C', 'G', 'S'))]
filter_df2 = filter_df2[['Category','Installs']]

top_categories2 = filter_df2.groupby('Category', as_index=False).sum().nlargest(5,"Installs").reset_index(drop=True)

top_categories2['Highlight'] = top_categories2['Installs'] > 1000000

#Figure 2
# Ensure the graph only displays between 6 PM to 8 PM
# current_time = dt.datetime.now().time()
if current_time >= dt.time(18, 0) and current_time <= dt.time(20, 0):
    fig2 = px.bar(top_categories2, x='Category', y='Installs', title='Top 5 App Categories by Global Installs',
                color='Highlight',color_discrete_map={True:'Green',False:'Blue'},
                width=1000,
                height=400
                
                )

    fig2.update_layout(
        legend_title='Exceeds 1 Million',
    )
else:
    # Create a blank figure with an annotation
    fig2 = go.Figure()
    fig2.add_annotation(
        text="The graph can only be displayed between 6 PM and 8 PM.",
        x=0.5, y=0.5, 
        showarrow=False,
        font=dict(size=20),
        xref='paper', yref='paper',
        align='center'
    )

    fig2.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=1000,
        height=400,
        title="Notice"
    )  
save_plot_as_html(fig2,"Task 2.html","")

# Get unique values in the 'Android Ver' column
unique_android_versions = apps_df['Android Ver'].unique()
unique_android_versions

# Function to parse Android version
def parse_android_version(version):
    if version == "Varies with device":
        return float('inf')
    elif '-' in version:
        return float(version.split('-')[0].strip().replace('.', '', 1))
    
    elif version == "4.4W and up":
        return float('4.4')

    else:
        return float(version.split()[0].replace('.', '', 1))
    
# Apply filters
filtered_df3 = apps_df[
    (apps_df['Installs'] >= 10000) &
    (apps_df['Revenue'] >= 10000) &
    (apps_df['Android Ver'].apply(parse_android_version) > 4.0)  &
    (apps_df['Size'] > 15) &
    (apps_df['Content Rating'] == 'Everyone') &
    (apps_df['App'].apply(lambda x: len(x) <= 30))
]

# Get top 3 categories
top_categories = filtered_df3['Category'].value_counts().nlargest(3).index

# Filter for top categories
filtered_df3 = filtered_df3[filtered_df3['Category'].isin(top_categories)]

# Calculate average installs and revenue for free vs paid apps within top categories
avg_data = filtered_df3.groupby(['Category', 'Type']).agg({'Installs': 'mean', 'Revenue': 'mean'}).reset_index()

# Ensure the graph only displays between 1 PM to 2 PM
# current_time = dt.datetime.now().time()
if current_time >= dt.time(13, 0) and current_time <= dt.time(14, 0):
# Plot dual-axis chart using plotly express
    fig3 = px.bar(avg_data, x='Category', y='Installs', color='Type',
                labels={'Installs': 'Average Installs'},
                title='Average Installs and Revenue for Free vs Paid Apps in Top Categories',
                width=1000,
                height=400)

    fig3.add_scatter(x=avg_data[avg_data['Type'] == 'Free']['Category'], 
                    y=avg_data[avg_data['Type'] == 'Free']['Revenue'], 
                    mode='lines+markers', name='Free Revenue')

    fig3.add_scatter(x=avg_data[avg_data['Type'] == 'Paid']['Category'], 
                    y=avg_data[avg_data['Type'] == 'Paid']['Revenue'], 
                    mode='lines+markers', name='Paid Revenue')

    fig3.update_layout(yaxis2=dict(title='Average Revenue',
                                overlaying='y',
                                side='right'))


else:
    # Create a blank figure with an annotation
    fig3 = go.Figure()
    fig3.add_annotation(
        text="The graph can only be displayed between 1 PM and 2 PM.",
        x=0.5, y=0.5, 
        showarrow=False,
        font=dict(size=20),
        xref='paper', yref='paper',
        align='center'
    )

    fig3.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=1000,
        height=400,
        title="Notice"
    )  
    
save_plot_as_html(fig3,"Task 3.html","")

# Create columns for month and year
apps_df['Last_Updated_Month'] = apps_df['Last Updated'].dt.month_name()
apps_df['Last_Updated_Year'] = apps_df['Last Updated'].dt.year

# Filter the data based on the specified conditions
filtered_df4 = apps_df[
    (apps_df['Rating'] >= 4.0) &  # Average rating >= 4.0
    (apps_df['Size'] >= 10) &  # Size >= 10M
    (apps_df['Last_Updated_Month'] == 'January')  # Last update in January
]

# Group the data by 'Category' and calculate the mean rating and total reviews
category_grouped = filtered_df4.groupby('Category').agg(
    Average_Rating=('Rating', 'mean'),
    Total_Reviews=('Reviews', 'sum'),
    Total_Installs=('Installs', 'sum')
).reset_index()

# Sort by installs and pick the top 10 categories
top_10_categories = category_grouped.sort_values(by='Total_Installs', ascending=False).head(10)

# Ensure the graph only displays between 3 PM to 5 PM
# current_time = dt.datetime.now().time()
if current_time >= dt.time(15, 0) and current_time <= dt.time(17, 0):
    # Create the figure
    fig4 = go.Figure()

    # Bar for Total Reviews
    fig4.add_trace(
        go.Bar(x=top_10_categories['Category'], y=top_10_categories['Total_Reviews'], 
               name="Total Reviews", yaxis='y', marker_color='orange')
    )

    # Line for Average Rating (scaled differently)
    fig4.add_trace(
        go.Scatter(x=top_10_categories['Category'], y=top_10_categories['Average_Rating'], 
                   name="Average Rating", yaxis='y2', marker=dict(color='blue'), mode='lines+markers')
    )

    # Update layout for dual y-axes
    fig4.update_layout(
        title="Average Rating and Total Review Count for Top 10 App Categories by Installs",
        xaxis=dict(title="Category"),
        yaxis=dict(title="Total Reviews", side='left'),
        yaxis2=dict(title="Average Rating", overlaying='y', side='right', range=[0, 5]),  # Scaling for rating (0-5)
        legend=dict(x=1.05, y=1, xanchor='left')
    )
    
else:
    # Create a blank figure with an annotation
    fig4 = go.Figure()
    fig4.add_annotation(
        text="Graph 4 can only be displayed between 3 PM and 5 PM.",
        x=0.5, y=0.5, 
        showarrow=False,
        font=dict(size=20),
        xref='paper', yref='paper',
        align='center'
    )

    fig4.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=1000,
        height=400,
        title="Notice"
    )  

save_plot_as_html(fig4,"Task 4.html","")

# Step 1: Filter apps updated within the year 2018
start_date = datetime(2018, 1, 1)
end_date = datetime(2018, 12, 31)
apps_df['Last Updated'] = pd.to_datetime(apps_df['Last Updated'], errors='coerce')  # Convert to datetime
filtered_df5 = apps_df[(apps_df['Last Updated'] >= start_date) & (apps_df['Last Updated'] <= end_date)]

# Convert 'Installs' and 'Reviews' to numeric, errors='coerce' will set non-numeric values to NaN
filtered_df5['Installs'] = pd.to_numeric(filtered_df5['Installs'], errors='coerce')
filtered_df5['Reviews'] = pd.to_numeric(filtered_df5['Reviews'], errors='coerce')

# Step 2: Filter apps with at least 100,000 installs and more than 1,000 reviews
filtered_df5 = filtered_df5[(filtered_df5['Installs'] >= 100000) & (filtered_df5['Reviews'] >= 1000)]


# Step 3: Filter out genres starting with specific characters
filtered_df5 = filtered_df5[~filtered_df5['Genres'].str.startswith(('A', 'F', 'E', 'G', 'I', 'K'))]

# Check if it's between 2 PM and 4 PM
# current_time = dt.datetime.now().time()
if current_time >= dt.time(14, 0) and current_time <= dt.time(16, 0):
    if not filtered_df5.empty:
        # Select relevant columns for the correlation matrix
        corr_data = filtered_df5[['Installs', 'Rating', 'Reviews']].corr()

        # Generate the heatmap with Plotly
        fig5 = px.imshow(
            corr_data,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            title="Correlation Matrix between Installs, Ratings, and Review Counts"
        )
        fig5.update_layout(width=600, height=500, title_x=0.5)

    else:
        # Create a blank figure with an annotation
        fig5 = go.Figure()
        fig5.add_annotation(
            text="No data available after applying the filters.",
            x=0.5, y=0.5, 
            showarrow=False,
            font=dict(size=20),
            xref='paper', yref='paper',
            align='center'
        )

        fig5.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            width=1000,
            height=400,
            title="Notice"
        )  
else:
    # Create a blank figure with an annotation
    fig5 = go.Figure()
    fig5.add_annotation(
        text="Graph 5 can only be displayed between 2 PM and 4 PM.",
        x=0.5, y=0.5, 
        showarrow=False,
        font=dict(size=20),
        xref='paper', yref='paper',
        align='center'
    )

    fig5.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=1000,
        height=400,
        title="Notice"
    )  

save_plot_as_html(fig5,"Task 5.html","")

#Filter apps containing 'C'
filtered_data = apps_df[apps_df['App'].str.contains('C', case=False)]

#Check category counts and filter categories with more than 50 apps
category_counts = filtered_data['Category'].value_counts()
valid_categories = category_counts[category_counts > 50].index
filtered_data = filtered_data[filtered_data['Category'].isin(valid_categories)]
    
#Filter apps with more than 10 reviews
filtered_data = filtered_data[filtered_data['Reviews'] >= 10]

#Filter apps with ratings less than 4.0
filtered_data = filtered_data[filtered_data['Rating'] < 4.0]

# Check if it's between 4 PM and 6 PM
# current_time = dt.datetime.now().time()
if current_time >= dt.time(16, 0) and current_time <= dt.time(18, 0):

    # Check if any data is available after filtering
    if not filtered_data.empty:
        # Create a violin plot to visualize the distribution of ratings
        fig6 = px.violin(filtered_data, y='Rating', x='Category', box=True, points='all')

        # Customize layout
        fig6.update_layout(
            title="Distribution of Ratings for Each App Category (Apps Containing 'C', Rating < 4.0)",
            yaxis_title="Rating",
            xaxis_title="Category",
            showlegend=False
        )

    else:
        print("No data available after applying the filters.")
else:
    # Create a blank figure with an annotation
    fig6 = go.Figure()
    fig6.add_annotation(
        text="Graph 6 can only be displayed between 4 PM and 6 PM.",
        x=0.5, y=0.5, 
        showarrow=False,
        font=dict(size=20),
        xref='paper', yref='paper',
        align='center'
    )

    fig6.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=1000,
        height=400,
        title="Notice"
    )  

save_plot_as_html(fig6,"Task 6.html","")

# Filter data based on the conditions
apps_df = apps_df[(apps_df['Content Rating'] == 'Teen') & (apps_df['Installs'] > 10000) & (apps_df['App'].str.startswith('E'))
]

# Create separate columns for month and year
apps_df['Year'] = apps_df['Last Updated'].dt.year
apps_df['Month'] = apps_df['Last Updated'].dt.month

# Group by category, year, and month and sum the installs
category_trends = apps_df.groupby(['Category', 'Year', 'Month'])['Installs'].sum().reset_index()

# Sort by date for calculating month-over-month growth
category_trends = category_trends.sort_values(['Category', 'Year', 'Month'])

# Calculate month-over-month percentage change
category_trends['MoM_Change'] = category_trends.groupby('Category')['Installs'].pct_change()

# Check if it's between 6 PM and 9 PM
# current_time = dt.datetime.now().time()
if current_time >= dt.time(18, 0) and current_time <= dt.time(21, 0):
    fig7 = go.Figure()

    # Plotting the line for each category and shading the regions where MoM_Change exceeds 20%
    categories = category_trends['Category'].unique()
    for category in categories:
        category_data = category_trends[category_trends['Category'] == category]
        
        # Plot the time series line for the category
        fig7.add_trace(go.Scatter(
            x=category_data['Year'].astype(str) + '-' + category_data['Month'].astype(str),
            y=category_data['Installs'],
            mode='lines',
            name=category,
            line=dict(width=2)
        ))

        # Filter rows where MoM_Change exceeds 20% for the current category
        significant_growth = category_data[category_data['MoM_Change'] > 0.2]
        if not significant_growth.empty:
            # Fill the area under the curve where growth exceeds 20%
            fig7.add_trace(go.Scatter(
                x=significant_growth['Year'].astype(str) + '-' + significant_growth['Month'].astype(str),
                y=significant_growth['Installs'],
                mode='none',
                fill='tozeroy',
                name=f'{category} >20% Growth',
                fillcolor='rgba(0,100,80,0.2)',  # Semi-transparent shading
                showlegend=True  # Showing legend for the shaded area
            ))
    # Customize the layout
    fig7.update_layout(
        title="Trend of Total Installs Over Time (Apps with 'Teen' Rating, Starting with 'E')",
        xaxis_title="Date",
        yaxis_title="Total Installs",
        legend_title="Category",
        hovermode="x unified",
        height=600
    )
    

else:
    # Create a blank figure with an annotation
    fig7 = go.Figure()
    fig7.add_annotation(
        text="Graph 7 can only be displayed between 6 PM and 9 PM.",
        x=0.5, y=0.5, 
        showarrow=False,
        font=dict(size=20),
        xref='paper', yref='paper',
        align='center'
    )

    fig7.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=1000,
        height=400,
        title="Notice"
    )


save_plot_as_html(fig7,"Task 7.html","")


plot_containers_split=plot_containers.split('</div>')

if len(plot_containers_split) > 1:
    final_plot=plot_containers_split[-2]+'</div>'
else:
    final_plot=plot_containers

dashboard_html= """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name=viewport" content="width=device-width,initial-scale-1.0">
    <title> Google Play Store Review Analytics</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #333;
            color: #fff;
            margin: 0;
            padding: 0;
        }}
        .header {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background-color: #444
        }}
        .header img {{
            margin: 0 10px;
            height: 50px;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            justify_content: center;
            padding: 20px;
        }}
        .plot-container {{
            border: 2px solid #555
            margin: 10px;
            padding: 10px;
            width: {plot_width}px;
            height: {plot_height}px;
            overflow: hidden;
            position: relative;
            cursor: pointer;
        }}
        .insights {{
            display: none;
            position: absolute;
            right: 10px;
            top: 10px;
            background-color: rgba(0,0,0,0.7);
            padding: 5px;
            border-radius: 5px;
            color: #fff;
        }}
        .plot-container: hover .insights {{
            display: block;
        }}
        </style>
        <script>
            function openPlot(filename) {{
                window.open(filename, '_blank');
                }}
        </script>
    </head>
    <body>
        <div class= "header">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Logo_2013_Google.png/800px-Logo_2013_Google.png" alt="Google Logo">
            <h1>Google Play Store Reviews Analytics</h1>
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Google_Play_Store_badge_EN.svg/1024px-Google_Play_Store_badge_EN.svg.png" alt="Google Play Store Logo">
        </div>
        <div class="container">
            {plots}
        </div>
    </body>
    </html>
    """

plot_width=1000
plot_height=600
plot_bg_color='black'
text_color='white'
title_font={'size':16}
axis_font={'size':12}

final_html=dashboard_html.format(plots=plot_containers,plot_width=plot_width,plot_height=plot_height)

dashboard_path=("index.html")

with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(final_html)

