import pandas as pd
import numpy as np
from plotly.offline import init_notebook_mode
init_notebook_mode(connected = True)
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import mannwhitneyu
import plotly.io as pio
from plotly.subplots import make_subplots
# %matplotlib inline


def calculate_mean_in_period(data, period, what):
    if 'during' not in period:
        mean_in_period = data.loc[period][what].mean()
    else:
        # In the case of the during-sale period, calculate the mean dividing by the (number of days -1)
        mean_in_period = data.loc[period][what].sum()/(len(data.loc[period])-1)
    return mean_in_period


def GUS_rel_ch(df, dates, recognition=False):
  sale_id = 'Sale 1' if dates['end'].year == 2017 else 'Sale 2'
  # fill missing ['Date', 'Country'] df with 0 in [BASIS]
  BASIS = 'Gross Units Sold'
  countries_u = df['Country'].unique()
  dates_u = df['Date'].unique()
  fill_missing = pd.MultiIndex.from_product([dates_u, countries_u, [0]], names=['Date', 'Country', BASIS])
  fill_missing = pd.DataFrame().reindex(fill_missing).reset_index()
  df = pd.concat([df, fill_missing])
  # add new column identifying period
  df['Period'] = 'before sale'  # initiate with 'before sale' status
  df.loc[df.Date >= dates['sale_start'], 'Period'] = 'during sale'
  df.loc[df.Date > dates['sale_end'], 'Period'] = 'after sale'
  # calculate global mean ratio
  grouped = df.groupby(['Period', 'Date']).sum().sort_index(level='Date')
  if not recognition:
    global_mean_ratio = calculate_mean_in_period(grouped, 'during sale', BASIS)
  else:
    global_mean_ratio = calculate_mean_in_period(grouped, 'after sale', BASIS)
  global_mean_ratio = (global_mean_ratio/calculate_mean_in_period(grouped, 'before sale', BASIS) - 1)*100
  # sum daily BASIS by period, date and country
  df = df[['Period', 'Date', 'Country', BASIS]].groupby(['Period', 'Date', 'Country']).sum().reset_index()
  # remove countries with abs(mean([BASIS]))<=0.1 in any named periods or less than 50 total [Net Units Sold]
  df['cond1'] = df[['Period', 'Country', BASIS]].groupby(['Period', 'Country']).transform('mean')
  cond1 = df[['Country', 'cond1']].groupby(['Country']).transform(lambda x: np.min(np.abs(x))) > 0.1
  df = df.drop(columns=['cond1'])
  cond2 = df[['Country', BASIS]].groupby(['Country']).transform('sum') > 50
  df = df[np.logical_and(cond1, cond2).values]
  
  percents = df.loc[df.Period == 'during sale']
  # Sum Gross Units Sold per country during the sale
  percents = percents[['Country', BASIS]].groupby(['Country']).sum().reset_index()
  total_BASIS = percents['Gross Units Sold'].sum()
  percents['percent'] = percents.apply(lambda r: r[BASIS] / total_BASIS, axis=1)
  percents = percents.drop_duplicates().values
  percents = {x[0]: x[2] for x in percents}
  
  if recognition:
    # estimate probability that before and after sale periods are different
    countries = df['Country'].unique()
    ps = {}
    for country in countries:
        nus_bs = df.loc[(df.Country == country) & (df.Period == 'before sale'), BASIS].values
        nus_as = df.loc[(df.Country == country) & (df.Period == 'after sale'), BASIS].values
        _, ps[country] = mannwhitneyu(nus_bs, nus_as)
    significant = [x for x in ps.keys() if ps[x]< 0.05]
    df = df.loc[df['Country'].isin(significant)]
    countries = df['Country'].unique()

  # calculate means per country in periods
  m_before_sale = df.loc[(df.Period == 'before sale'), ['Country', BASIS]].groupby(['Country']).mean()
  m_before_sale = m_before_sale.reset_index().values
  m_before_sale = {x[0]: x[1] for x in m_before_sale}
  # calculate mean during sale by dividing by (number of days - 1),
  # because during the first and last day of the sale, the promotion takes only a fraction of the day
  # for example (19-24) on the first day and (0-19) on the last day
  m_sale = (df.loc[(df.Period == 'during sale'), ['Country', BASIS]].groupby(['Country'])
            .agg(lambda x: np.sum(x)/(len(x)-1)))
  m_sale = m_sale.reset_index().values
  m_sale = {x[0]: x[1] for x in m_sale}
  m_after_sale = df.loc[(df.Period == 'after sale'), ['Country', BASIS]].groupby(['Country']).mean()
  m_after_sale = m_after_sale.reset_index().values
  m_after_sale = {x[0]: x[1] for x in m_after_sale}
  # calculate means ratios and sort descending
  if not recognition:
    df['norm '+BASIS+' means ratio'] = df.apply(
        lambda r: m_sale[r['Country']] / m_before_sale[r['Country']] -1, axis=1)
    df = df.sort_values(by=['norm '+BASIS+' means ratio'], ascending=False)
  else:
    df['norm '+BASIS+' means ratio'] = df.apply(
        lambda r: m_after_sale[r['Country']] / m_before_sale[r['Country']] -1, axis=1)
    df = df.sort_values(by=['norm '+BASIS+' means ratio'], ascending=False)
  
  countries = df['Country'].unique()
  # get ratios between before and during sale period means
  nNUSmRatio = df[['Country', 'norm '+BASIS+' means ratio']].drop_duplicates().values
  nNUSmRatio = {x[0]: x[1]*100 for x in nNUSmRatio}

  # plot normalized GUS data
  fig_in = go.Figure()
  fig_in.add_trace(go.Scatter(x=countries, y=[nNUSmRatio[c] for c in countries], mode='markers',
                              name='Average daily sales increase', showlegend=False, hovertemplate='%{x}: %{y:.0f}',
                              marker_symbol='line-ew', marker_line_color='darkgreen',
                              marker_color='green', marker_line_width=2, marker_size=10))
  fig_in.add_trace(go.Scatter(x=countries, y=[nNUSmRatio[c] for c in countries], mode='lines',
                              name='', showlegend=False, hovertemplate='', opacity=0.3,
                              line_color='black', line_dash='dot'))
  fig_in.add_trace(go.Scatter(x=countries, y=[global_mean_ratio]*len(countries), mode='lines',
                              name='Global average', showlegend=False, hovertemplate='%{y:.0f}', opacity=0.3,
                              line_color='green', line_dash='dash'))
  # details: add alternating background and ratio values
  for country in countries:
      i = np.where(np.array([*nNUSmRatio]) == country)[0][0]
      # colors = ['LightSkyBlue', 'azure']
      colors = ['darkorchid', 'orange']
      fig_in.add_shape(
          type='rect',
          xref='x',
          yref='paper',
          x0=i - 0.5,
          y0=0,
          x1=i + 0.5,
          y1=percents[country],
          opacity=0.2,
          line=dict(width=0),
          fillcolor=colors[0] if i % 2 else colors[1])
      fig_in.add_shape(
          type='rect',
          xref='x',
          yref='paper',
          x0=i - 0.5,
          y0=percents[country],
          x1=i + 0.5,
          y1=1,
          opacity=0.05,
          line=dict(width=0),
          fillcolor=colors[0] if i % 2 else colors[1])
  fig_in.update_layout(
      title_text='Average daily sales increase <b>' + 
                 ('after' if recognition else 'during') + 
                 ' sale</b> per <b>country</b>',
      title_x=0.5,
      title_y=0.9,
      height=600,
      yaxis_title_text='Average daily sales increase [%]',
      legend_title_text=None,
  )
  plt.plot([nNUSmRatio[c] for c in countries])
  plt.gca().set_ylim([0, None])
  ticks = np.sort(np.append(plt.gca().get_yticks(), [round(global_mean_ratio)]))
  plt.close('all')
  fig_in.update_yaxes(range=[ticks[0], ticks[-1]])
  fig_in.update_yaxes(tickvals=ticks)
  fig_in.show()
  
  
def choropleth(sales_df, country_codes, pop_data):
  sale_id = 'Sale 1' if sales_df.Date.max().year == 2017 else 'Sale 2'
  template = pio.templates.default
  pio.templates.default = 'plotly'
  # Take only the during-sale data
  s_during = sales_df.loc[(sales_df.Period == 'during sale')]
  # Sum Gross Units Sold per country code during the sale
  s_during = s_during[['Country Code', 'Gross Units Sold']].groupby(['Country Code']).sum().reset_index()
  s_during = pd.merge(s_during, country_codes, left_on='Country Code', right_on='Alpha2', how='left')
  s_during = s_during.drop(columns=['Alpha2', 'Country Code'])
  s_during = pd.merge(s_during, pop_data, left_on='Alpha3', right_on='Country Code', how='left')
  s_during['Gross Units Sold per 1M pop.'] = s_during.apply(
      lambda r: (r['Gross Units Sold']/((r['2017'] + r['2018']) / 2))*1e6,
      axis=1
  )
  # draw the map for absolute Gross Units Sold
  fig = go.Figure(data=go.Choropleth(
    locations=s_during['Alpha3'],
    z=np.log10(s_during['Gross Units Sold']),
    text=s_during['Country'],
    customdata=s_during['Gross Units Sold'].values,
    hovertemplate='<b>%{text}</b><br>Gross Units Sold: %{customdata}',
    name='',
    colorbar_title='',
    colorbar_tickvals=[0, 1, 2, 3, 4, 5],
    colorbar_ticktext=["1", "10", "100", "1K", '10k', '100k'],
    colorscale='viridis',
    marker_line_color='navy',
    marker_line_width=0.5,
  ))
  # draw the map for Gross Units Sold per 1M pop.
  map_norm = go.Choropleth(
      locations=s_during['Alpha3'],
      z=np.log10(s_during['Gross Units Sold per 1M pop.']),
      text=s_during['Country'],
      customdata=s_during['Gross Units Sold per 1M pop.'].values,
      hovertemplate='<b>%{text}</b><br>Gross Units Sold per 1M pop.: %{customdata:.1f}',
      name='',
      colorbar_title='',
      colorbar_tickvals=[0, 1, 2, 3],
      colorbar_ticktext=['1', '10', '100', '1K'],
      colorscale='viridis',
      marker_line_color='navy',
      marker_line_width=0.5,
  )
  map_norm['visible'] = False
  fig.add_trace(map_norm)

  fig.update_layout(
      title_text=f'<b>Gross Units Sold</b> during <b>{sale_id}</b>',
      title_x=0.5,
      geo=dict(
          showframe=False,
          showcoastlines=False,
          projection_type='equirectangular'
      ),
      updatemenus=[
          dict(
              type='buttons',
              direction='left',
              buttons=list([
                  dict(
                      args=[{'visible': [True] * 1 + [False] * 1}],
                      label='Gross Units Sold',
                      method='update'
                  ),
                  dict(
                      args=[{'visible': [False] * 1 + [True] * 1}],
                      label='Gross Units Sold per 1M pop.',
                      method='update'
                  ),
              ]),
              pad={'r': 10, 't': 10},
              showactive=True,
              x=0.0,
              xanchor='left',
              y=1.01,
              yanchor='bottom'
          ),
      ]
  )
  fig.show()
  pio.templates.default = template
  
def plot_gauge1(fig_in, means_sale1, what):
    col = 0 if what == 'Gross Units Sold' else 1
    plt.plot([means_sale1['before sale'], means_sale1['during sale']])
    plt.gca().set_ylim([0, None])
    tickvals1 = np.sort(np.append(plt.gca().get_yticks(),
                                  [means_sale1['before sale'], means_sale1['during sale']]
                                 ))
    plt.close('all')
    ticknames = {means_sale1['before sale']: '<b>before Sale</b>', means_sale1['during sale']: '<b>during Sale</b>'}
    ticktext1 = [f'{x:.0f}' if x not in ticknames else ticknames[x] for x in tickvals1]
    gauge_axis = dict(range=[None, tickvals1[-1]],
                      tickmode='array',
                      tickangle=0,
                      tickvals=tickvals1,
                      ticktext=ticktext1
                     )
    # indicate the before-sale average as a reference level with a gray field
    gauge_Steps = [{'range': [0, means_sale1['before sale']], 'color': "lightgray"}]
    fig_in.add_trace(go.Indicator(
        mode='gauge+number+delta',  # draw the gauge, achieved value and it's change
        value=means_sale1['during sale'],
        gauge_steps=gauge_Steps,
        gauge_axis=gauge_axis,
        # Set the reference value to the before-sale mean and delta-mode to relative (shows relative change in %)
        delta={'reference': means_sale1['before sale'], 'relative': True},
        title={'text': 'mean daily <b>Gross Units Sold</b><BR>(<b>Sale 1</b>)'},
        domain={'y': [0.0, 1], 'x': [col*0.5+0.5*0.3, col*0.5+0.5*0.7]},
    ))


def plot_gauge2(fig_in, means_sale1, means_sale2, what):
    col = 0 if what == 'Gross Units Sold' else 1
    plt.plot([means_sale1['before sale'],
              means_sale1['during sale'],
              means_sale2['before sale'],
              means_sale2['during sale'],
             ])
    plt.gca().set_ylim([0, None])
    tickvals1 = np.sort(np.append(plt.gca().get_yticks(),
                                  [means_sale1['before sale'], means_sale1['during sale']]
                                 ))
    tickvals2 = np.sort(np.append(tickvals1,
                                  [means_sale2['before sale'], means_sale2['during sale']]
                                 ))
    plt.close('all')
    ticknames = {means_sale1['before sale']: '',
                 means_sale1['during sale']: 'during Sale 1',
                 means_sale2['before sale']: '<b>before Sale 2</b><BR>before Sale 1',
                 means_sale2['during sale']: '<b>during Sale 2<BR></b>'
                }
    ticktext2 = [f'{x:.0f}' if x not in ticknames else ticknames[x] for x in tickvals2]
    gauge_axis = dict(range=[None, tickvals2[-1]],
                      tickmode='array',
                      tickangle=0,
                      tickvals=tickvals2,
                      ticktext=ticktext2
                     )
    gauge_Steps = [{'range': [0, means_sale1['before sale']], 'color': "gray"},
                   {'range': [means_sale1['before sale'], means_sale2['before sale']], 'color': "lightgray"},
                   {'range': [means_sale2['before sale'], means_sale1['during sale']], 'color': "darkseagreen"}]
    fig_in.add_trace(go.Indicator(
        mode='gauge+number+delta',  # draw the gauge, achieved value and it's change
        value=means_sale2['during sale'],
        gauge_steps=gauge_Steps,
        gauge_axis=gauge_axis,
        # Set the reference value to the before-sale mean and delta-mode to relative (shows relative change in %)
        delta={'reference': means_sale2['before sale'], 'relative': True},
        title={'text': 'mean daily <b>' + what + '</b><BR>(<b>Sale 2</b>)'},
        domain={'y': [0.2, 1], 'x': [col*0.5+0.5*0.3, col*0.5+0.5*0.7]},
    ))
    fig_in.add_trace(go.Indicator(
        mode='delta',
        value=(means_sale2['during sale']-means_sale2['before sale'])/means_sale2['before sale'],
        delta={'reference': (means_sale1['during sale']-means_sale1['before sale'])/means_sale1['before sale'],
               'relative': True
              },
        title={'text': '<b>Sale 2</b> vs <b>Sale 1</b>'},
        domain={'y': [0.2, 0.4], 'x': [col*0.5+0.5*0.4, col*0.5+0.5*0.6]},
    ))

  
def story_plots():
  data = pd.read_excel('sales.xlsx', parse_dates=True)
  data.drop(labels=['Package', 'Product(ID#)', 'Product Name', 'Type'], axis='columns', inplace=True)
  data['discount'] = data.apply(
    lambda r: round(100 - (r['Sale Price'] / r['Base Price']) * 100),
    axis=1
  )
  sales = {1: data[data['Date'] < '07-2017'].copy(), 2: data[data['Date'] > '07-2017'].copy()}
  # sale 1 cleaning
  discounts = sales[1][['discount', 'Gross Units Sold']].groupby(['discount']).sum()
  discounts.sort_values('Gross Units Sold', ascending=False, inplace=True)
  # discounts.index holds 'discount' data and discounts.values contains GUS data
  discounts = [disc for disc, GUS in zip(discounts.index, discounts.values) if GUS>0]
  dates_sale1 = {'start': sales[1]['Date'].min(),
                 'sale_start': pd.Timestamp('2016-12-22 00:00:00'),
                 'sale_end': pd.Timestamp('2017-01-02 00:00:00'),
                 'end': sales[1]['Date'].max()}
  sales[1]['Period'] = sales[1].apply(lambda r: 'before sale' if r['Date']<dates_sale1['sale_start'] else
                                        'after sale' if r['Date']>dates_sale1['sale_end'] else
                                        'during sale', axis=1)
  sales[1] = sales[1].set_index(['Period', 'discount'])
  sales[1] = sales[1].loc[[('before sale', 0), ('during sale', discounts[0]), ('after sale', 0)]].reset_index()
  s1_grouped = sales[1].groupby(['Period', 'Date']).sum().sort_index(level='Date')

  # sale 2 cleaning
  sales[2]['discount'] = sales[2].apply(lambda r: r['discount'] if r['discount'] != 42 else 40, axis=1)
  discounts1 = discounts
  discounts = sales[2][['discount', 'Gross Units Sold']].groupby(['discount']).sum()
  discounts.sort_values('Gross Units Sold', ascending=False, inplace=True)
  # discounts.index holds 'discount' data and discounts.values contains GUS data
  discounts = [disc for disc, GUS in zip(discounts.index, discounts.values) if GUS>0]
  dates_sale2 = {'start': sales[2]['Date'].min(),
                 'sale_start': pd.Timestamp('2017-12-21 00:00:00'),
                 'sale_end': pd.Timestamp('2018-01-03 00:00:00'),
                 'end': sales[2]['Date'].max()}
  sales[2]['Period'] = sales[2].apply(lambda r: 'before sale' if r['Date']<dates_sale2['sale_start'] else
                                        'after sale' if r['Date']>dates_sale2['sale_end'] else
                                        'during sale', axis=1)
  sales[2] = sales[2].set_index(['Period', 'discount'])
  sales[2] = sales[2].loc[[('before sale', 0), ('during sale', discounts[0]), ('after sale', 0)]].reset_index()
  s2_grouped = sales[2].groupby(['Period', 'Date']).sum().sort_index(level='Date')

  pio.templates.default = 'plotly_white'
  
  # ===========================================================================
  # ================================ PLOT 1 ===================================
  # ===========================================================================
  fig = go.Figure()
  # plot the total Gross Units Sold per day
  fig.add_trace(go.Scatter(x=s1_grouped.index.get_level_values('Date'),
                           y=s1_grouped['Gross Units Sold'],
                           mode='lines+markers',
                           name='Gross Units Sold',
                           line=dict(color='gray', dash=None, width=4),
                           hovertemplate='%{y:.0f}',
                           marker_size=8
                          ))
  # set the colors for before-sale, during-sale and after-sale means
  colors = {'before sale': 'dodgerblue', 'during sale': 'mediumseagreen', 'after sale': 'lightcoral'}
  means_sale1 = {}
  means_sale1_GS = {}
  for period in ['before sale', 'during sale', 'after sale']:
      means_sale1[period] = mean_in_period = calculate_mean_in_period(s1_grouped, period, 'Gross Units Sold')
      means_sale1_GS[period] = calculate_mean_in_period(s1_grouped, period, 'Gross Sales (USD)')
      fig.add_trace(go.Scatter(x=s1_grouped.loc[period].index.get_level_values('Date'),
                               y=[mean_in_period] * len(s1_grouped.loc[period]),
                               mode='lines',
                               name='daily average ' + period,
                               line=dict(color=colors[period],dash=None, width=4),
                               marker_size=4,
                               opacity=0.7,
                               hovertemplate='%{y:.0f}'
                              ))
  span_shape = dict(type='rect',
                    xref='x',
                    yref='paper',
                    x0=dates_sale1['sale_start'],
                    y0=0,
                    x1=dates_sale1['sale_end'],
                    y1=1,
                    opacity=0.1,
                    line=dict(width=0),
                    fillcolor='green'
                   )
  fig.update_layout(
      legend=dict(
        yanchor='top',
        y=0.99,
        xanchor='right',
        x=0.99
        ),
      shapes=[span_shape],
      title_text='<b>Gross daily sales</b> worldwide before, during and after <b>Sale 1</b>',
      title_x=0.5,
      yaxis_title_text='Gross Units Sold',
      xaxis_title_text='Date',
      hovermode="x unified",
      )
  fig.update_xaxes(tickvals=[dates_sale1['start'], dates_sale1['sale_start'], dates_sale1['sale_end'], dates_sale1['end']])
  fig.add_annotation(text=f'{discounts1[0]}% off',
                     y=0.0,
                     opacity=0.3,
                     x=dates_sale1['sale_start'] + (dates_sale1['sale_end'] - dates_sale1['sale_start']) / 2,
                     yref='paper',
                     align='left',
                     textangle=-30,
                     showarrow=False,
                     font=dict(color="darkgreen", size=30)
                    )
  fig.show()
  yield
  # ===========================================================================
  # ================================ PLOT 2 ===================================
  # ===========================================================================  
  country_codes = pd.read_excel('countries.xlsx')
  pop_data = pd.read_excel('populations.xlsx', usecols=lambda x: x in ('Country Code', '2016', '2017', '2018'))
  choropleth(sales[1], country_codes, pop_data)
  yield
  # ===========================================================================
  # ================================ PLOT 3 ===================================
  # ===========================================================================
  GUS_rel_ch(sales[1], dates_sale1)
  yield
  # ===========================================================================
  # ================================ PLOT 4 ===================================
  # ===========================================================================
  fig = go.Figure()
  plot_gauge1(fig, means_sale1, 'Gross Units Sold')
  plot_gauge1(fig, means_sale1_GS, 'Gross Sales (USD)')
  fig.update_layout(width=900, height=300)
  fig.show()
  yield
  # ===========================================================================
  # ================================ PLOT 5 ===================================
  # ===========================================================================
  GUS_rel_ch(sales[1], dates_sale1, True)
  yield  
  # ===========================================================================
  # ================================ PLOT 6 ===================================
  # ===========================================================================
  fig = make_subplots(rows=1,
    cols=2,
    shared_yaxes=True,
    subplot_titles=('Sale 1', '<b>Sale 2</b>'),
    horizontal_spacing=0.02
  )
  # plot the total Gross Units Sold per day  
  main_discount = {1: discounts1[0], 2: discounts[0]}
  dates_sale = {1: dates_sale1, 2: dates_sale2}
  s_grouped = {1: s1_grouped, 2: s2_grouped}
  for sale_n in [1, 2]:
    fig.add_trace(go.Scatter(x=s_grouped[sale_n].index.get_level_values('Date'),
                             y=s_grouped[sale_n]['Gross Units Sold'],
                             mode='lines+markers',
                             name='Gross Units Sold',
                             showlegend=sale_n == 2,
                             line=dict(color='gray', dash=None, width=2),
                             hovertemplate='%{y:.0f}',
                             marker_size=5,
                             opacity=1*sale_n/2,
                             # opacity=1*sale_n**2/4,
                            ),
                  row=1, col=sale_n
                 )
    # set the colors for before-sale, during-sale and after-sale means
    colors = {'before sale': 'dodgerblue', 'during sale': 'mediumseagreen', 'after sale': 'lightcoral'}
    means_sale2 = {}
    means_sale2_GS = {}
    for period in ['before sale', 'during sale', 'after sale']:
        means_sale2[period] =  mean_in_period = calculate_mean_in_period(s_grouped[sale_n], period, 'Gross Units Sold')
        means_sale2_GS[period] = calculate_mean_in_period(s2_grouped, period, 'Gross Sales (USD)')
        fig.add_trace(go.Scatter(x=s_grouped[sale_n].loc[period].index.get_level_values('Date'),
                                 y=[mean_in_period] * len(s_grouped[sale_n].loc[period]),
                                 mode='lines',
                                 name='daily average ' + period,
                                 showlegend=sale_n == 2,
                                 line=dict(color=colors[period],dash=None, width=3),
                                 marker_size=4,
                                 opacity=0.7*sale_n/2,
                                 # opacity=0.7*sale_n**2/4,
                                 hovertemplate='%{y:.0f}'
                                ),
                      row=1, col=sale_n
                     )
    fig.add_shape(dict(type='rect',
                      xref=f'x{sale_n}',
                      yref='paper',
                      x0=dates_sale[sale_n]['sale_start'],
                      y0=0,
                      x1=dates_sale[sale_n]['sale_end'],
                      y1=1,
                      opacity=0.1*sale_n/2,
                      # opacity=0.1*sale_n**2/4,
                      line=dict(width=0),
                      fillcolor='green'
                     ))
    fig.update_layout(
        legend=dict(
          yanchor='top',
          y=0.99,
          xanchor='center',
          x=0.5
          ),
        # shapes=[span_shape],
        title_text='<b>Gross daily sales</b> worldwide before, during and after sales',
        title_x=0.5,
        yaxis_title_text='Gross Units Sold',
        xaxis_title_text='Date',
        hovermode="x unified",
        )
    fig.update_xaxes(tickvals=[dates_sale[sale_n]['start'], dates_sale[sale_n]['sale_start'], dates_sale[sale_n]['sale_end'], dates_sale[sale_n]['end']], row=1, col=sale_n)
    fig.add_annotation(text=f'{main_discount[sale_n]}% off',
                       y=0.0,
                       opacity=0.3*sale_n/2,
                       # opacity=0.3*sale_n**2/4,
                       x=dates_sale[sale_n]['sale_start'] + (dates_sale[sale_n]['sale_end'] - dates_sale[sale_n]['sale_start']) / 2,
                       xref=f'x{sale_n}',
                       yref='paper',
                       align='left',
                       textangle=-30,
                       showarrow=False,
                       font=dict(color="darkgreen", size=20)
                      )
  fig.show()
  yield
  # ===========================================================================
  # ================================ PLOT 7 ===================================
  # ===========================================================================  
  choropleth(sales[2], country_codes, pop_data)
  yield
  # ===========================================================================
  # ================================ PLOT 8 ===================================
  # ===========================================================================
  GUS_rel_ch(sales[2], dates_sale2)
  yield
  # ===========================================================================
  # ================================ PLOT 9 ===================================
  # ===========================================================================
  fig = go.Figure()
  plot_gauge2(fig, means_sale1, means_sale2, 'Gross Units Sold')
  plot_gauge2(fig, means_sale1_GS, means_sale2_GS, 'Gross Sales (USD)')
  fig.update_layout(width=900, height=500)
  fig.show()
  yield
  # ===========================================================================
  # ================================ PLOT 10 ===================================
  # ===========================================================================
  GUS_rel_ch(sales[2], dates_sale2, True)
  yield
  
  
def depict_sample(seed=1):
    np.random.seed(seed)
    unaware = np.array([[0, 0]])
    for i in range(10000):
        new = np.random.rand(2)
        if np.all(np.sqrt(np.sum((unaware-new)**2, axis=1)) > 0.05):
            unaware = np.vstack([unaware, new])
    unaware += 1000
    ind = np.random.choice(len(unaware), size=[int(len(unaware)*0.02)], replace=False)
    ind_inversed = list(set(range(len(unaware))) - set(ind))
    owns = unaware[ind]
    unaware = unaware[ind_inversed]

    ind = np.random.choice(len(unaware), size=[int(len(unaware)*0.03)], replace=False)
    ind_inversed = list(set(range(len(unaware))) - set(ind))
    considering = unaware[ind]
    unaware = unaware[ind_inversed]

    ind = np.random.choice(len(unaware), size=[int(len(unaware)*0.01)], replace=False)
    ind_inversed = list(set(range(len(unaware))) - set(ind))
    not_interested = unaware[ind]
    unaware = unaware[ind_inversed]

    inds = np.zeros(len(unaware))
    for point in owns:
        inds += np.sqrt(np.sum((unaware-point)**2, axis=1)) < 0.1
    for point in considering:
        inds += np.sqrt(np.sum((unaware-point)**2, axis=1)) < 0.05
    inds = inds.astype(bool)
    considering = np.vstack([considering, unaware[inds]])
    unaware = unaware[~inds]

    ms = 9
    lw = 1
    template = pio.templates.default
    pio.templates.default = 'plotly_white'


    fig = make_subplots(rows=1,
                        cols=4,
                        subplot_titles=('before <b>Sale 1</b>',
                                        'after <b>Sale 1</b>',
                                        'before <b>Sale 2</b>',
                                        'after <b>Sale 2</b>'
                                        ),
                        horizontal_spacing=0.001
                       )


    def plot_situation(fig_in, u, c, ni, o, col=1):
        fig.add_trace(go.Scatter(x=u[:,0],
                                 y=u[:,1],
                                 marker_color='silver',
                                 mode='markers',
                                 marker_size=ms,
                                 name='unaware of the book\'s existence',
                                 hovertemplate='unaware of the book\'s existence<extra></extra>',
                                 marker_line_width=lw,
                                 showlegend=col == 1,
                                ),
                     row=1, col=col
                     )
        fig.add_trace(go.Scatter(x=c[:,0],
                                 y=c[:,1],
                                 marker_color='lemonchiffon',
                                 mode='markers',
                                 marker_size=ms,
                                 name='considers buying',
                                 hovertemplate='considers buying<extra></extra>',
                                 marker_line_width=lw,
                                 showlegend=col == 1,
                                ),
                     row=1, col=col
                     )
        fig.add_trace(go.Scatter(x=ni[:,0],
                                 y=ni[:,1],
                                 marker_color='lightcoral',
                                 mode='markers',
                                 marker_size=ms,
                                 name='not interested',
                                 hovertemplate='not interested<extra></extra>',
                                 marker_line_width=lw,
                                 showlegend=col == 1,
                                ),
                     row=1, col=col
                     )
        fig.add_trace(go.Scatter(x=o[:,0],
                                 y=o[:,1],
                                 marker_color='lightgreen',
                                 mode='markers',
                                 marker_size=ms,
                                 name='owns the book',
                                 hovertemplate='owns the book<extra></extra>',
                                 marker_line_width=lw,
                                 showlegend=col == 1,
                                ),
                     row=1, col=col
                     )
        fig.update_xaxes(showgrid=False, showticklabels=False)#, row=2, col=1)
        fig.update_yaxes(showgrid=False, showticklabels=False)#, row=2, col=1)
        
        
    plot_situation(fig, unaware, considering, not_interested, owns)
    
    
    # after Sale 1:
    def evolve(unaware, considering, owns, not_interested, p1, p2, p3, p4, r1, r2):
      ind = np.random.choice(len(considering), size=[int(len(considering)*p1)], replace=False)
      ind_inversed = list(set(range(len(considering))) - set(ind))
      owns = np.vstack([owns, considering[ind]])
      considering = considering[ind_inversed]

      ind = np.random.choice(len(unaware), size=[int(len(unaware)*p2)], replace=False)
      ind_inversed = list(set(range(len(unaware))) - set(ind))
      owns = np.vstack([owns, unaware[ind]])
      unaware = unaware[ind_inversed]

      ind = np.random.choice(len(unaware), size=[int(len(unaware)*p3)], replace=False)
      ind_inversed = list(set(range(len(unaware))) - set(ind))
      considering = np.vstack([considering, unaware[ind]])
      unaware = unaware[ind_inversed]

      ind = np.random.choice(len(unaware), size=[int(len(unaware)*p4)], replace=False)
      ind_inversed = list(set(range(len(unaware))) - set(ind))
      not_interested = np.vstack([not_interested, unaware[ind]])
      unaware = unaware[ind_inversed]

      inds = np.zeros(len(unaware))
      for point in owns:
          inds += np.sqrt(np.sum((unaware-point)**2, axis=1)) < r1
      for point in considering:
          inds += np.sqrt(np.sum((unaware-point)**2, axis=1)) < r2
      inds = inds.astype(bool)
      considering = np.vstack([considering, unaware[inds]])
      unaware = unaware[~inds]
      return unaware, considering, owns, not_interested


    unaware, considering, owns, not_interested = (
      evolve(unaware, considering, owns, not_interested, 0.7, 0.1, 0.05, 0.03, 0.1, 0.05)
      # evolve(unaware, considering, owns, not_interested, 0.6, 0.1, 0.05, 0.03, 0.1, 0.05)
    )
    plot_situation(fig, unaware, considering, not_interested, owns, 2)

    # before Sale 2:
    unaware, considering, owns, not_interested = (
      # evolve(unaware, considering, owns, not_interested, 0.1, 0.01, 0.3, 0.01, 0.1, 0.05)
      evolve(unaware, considering, owns, not_interested, 0.1, 0.01, 0.1, 0.01, 0.1, 0.05)
    )
    plot_situation(fig, unaware, considering, not_interested, owns, 3)

    # after Sale 2:
    unaware, considering, owns, not_interested = (
      # evolve(unaware, considering, owns, not_interested, 0.4, 0.1, 0.1, 0.1, 0.05, 0.02)
      evolve(unaware, considering, owns, not_interested, 0.3, 0.1, 0.1, 0.1, 0.05, 0.02)
    )
    plot_situation(fig, unaware, considering, not_interested, owns, 4)

    fig.update_layout(
        legend=dict(
            orientation='h',
            yanchor='top',
            y=0,
            xanchor='left',
            x=0
        ),
        height=400
    )
    fig.show()
    pio.templates.default = template