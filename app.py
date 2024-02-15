import fastf1 as ff1
import streamlit as st
import matplotlib as mpl
from fastf1 import plotting
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
import seaborn as sns
from fastf1.ergast import Ergast
import plotly.express as px
from plotly.io import show
from matplotlib import cm

ff1.plotting.setup_mpl(misc_mpl_mods=False)

def laptimes_scatter(year,gpname):


  session = ff1.get_session(int(year),str(gpname), 'R')
  session.load(telemetry = False,weather=False)
  session.laps.sort_values("Position",inplace = True)
  
  drivers = session.laps['Driver'].unique()

  # Create a 4x5 grid of plots
  fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(20, 16))

  # Loop through all the drivers and create a separate plot for each driver
  for i, driver in enumerate(drivers):
      # Retrieve the lap times and tyre compounds of the driver
      driver_laps = session.laps.pick_driver(driver).pick_quicklaps().reset_index()
      

      # Create a scatter plot of the lap times
      row = i // 5
      col = i % 5
      sns.scatterplot(data=driver_laps,
                      x="LapNumber",
                      y="LapTime",
                      ax=axs[row, col],
                      hue="Compound",
                      palette=ff1.plotting.COMPOUND_COLORS,
                      s=80,
                      linewidth=0,
                      legend='auto')



      axs[row, col].set_title(driver)
      axs[row, col].set_xlabel('Lap Number')
      axs[row, col].set_ylabel('Lap Time (s)')

  plt.tight_layout()
  return plt.gcf()
  # plt.savefig("racers lap time vs lap number.png")
  # plt.show()

def results(year):

  ergast = Ergast()
  races = ergast.get_race_schedule(year)  
  results = []

  # For each race in the season
  for rnd, race in races['raceName'].items():

      
      temp = ergast.get_race_results(season=int(year), round=rnd + 1)
      temp = temp.content[0]

      # If there is a sprint, get the results as well
      sprint = ergast.get_sprint_results(season=int(year), round=rnd + 1)
      if sprint.content and sprint.description['round'][0] == rnd + 1:
          temp = pd.merge(temp, sprint.content[0], on='driverCode', how='left')
          # Add sprint points and race points to get the total
          temp['points'] = temp['points_x'] + temp['points_y']
          temp.drop(columns=['points_x', 'points_y'], inplace=True)

      # Add round no. and grand prix name
      temp['round'] = rnd + 1
      temp['race'] = race.removesuffix(' Grand Prix')
      temp = temp[['round', 'race', 'driverCode', 'points']] 
      results.append(temp)

  
  results = pd.concat(results)
  races = results['race'].drop_duplicates()
  results = results.pivot(index='driverCode', columns='round', values='points')
  

  # Rank the drivers by their total points
  results['total_points'] = results.sum(axis=1)
  results = results.sort_values(by='total_points', ascending=False)
  results.drop(columns='total_points', inplace=True)

  # Use race name, instead of round no., as column names
  results.columns = races
  
  fig = px.imshow(
      results,
      text_auto=True,
      aspect='auto',  
      color_continuous_scale=[[0,    'rgb(198, 219, 239)'],  # Blue scale
                              [0.25, 'rgb(107, 174, 214)'],
                              [0.5,  'rgb(33,  113, 181)'],
                              [0.75, 'rgb(8,   81,  156)'],
                              [1,    'rgb(8,   48,  107)']],
      labels={'x': 'Race',
              'y': 'Driver',
              'color': 'Points'}       # Change hover texts
  )
  fig.update_xaxes(title_text='')      # Remove axis titles
  fig.update_yaxes(title_text='')
  fig.update_yaxes(tickmode='linear')  # Show all ticks, i.e. driver names
  fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey',
                  showline=False,
                  tickson='boundaries')              # Show horizontal grid only
  fig.update_xaxes(showgrid=False, showline=False)    # And remove vertical grid
  fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')     # White background
  fig.update_layout(coloraxis_showscale=False)        # Remove legend
  fig.update_layout(xaxis=dict(side='top'))           # x-axis on top
  fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))  # Remove border margins
  return fig
  #show(fig)

def driver_speeds(year,gpname,p1):
  # session = ff1.get_session(int(year), str(gpname), 'R')
  # session.load()
  p1_name =str(p1)
  driver = p1_name
  
  colormap = cm.get_cmap('plasma')
  session.load()
  lap = session.laps.pick_driver(driver).pick_fastest()

 # telemetry data
  x = lap.telemetry['X']            
  y = lap.telemetry['Y']             
  color = lap.telemetry['Speed']      

  points = np.array([x, y]).T.reshape(-1, 1, 2)
  segments = np.concatenate([points[:-1], points[1:]], axis=1)

  fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(12, 6.75))
  # fig.set_facecolor('grey')
  fig.suptitle(f'{gp_name} {year} - {driver} - Speed', size=16, y=0.97)  

  plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.12)
  ax.axis('off')

  plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.12)
  ax.axis('off')

  # Create background track line
  ax.plot(lap.telemetry['X'], lap.telemetry['Y'], color='gray', linestyle='-', linewidth=6, zorder=0,alpha =0.3)

  # Create a continuous norm to map from data points to colors
  norm = plt.Normalize(1, colormap.N+1)
  lc = LineCollection(segments, cmap=colormap, norm=norm, linestyle='-', linewidth=4)



  plt.gca().add_collection(lc)
  plt.axis('equal')
  plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

  
  lc.set_array(color)

  
  line = ax.add_collection(lc)


  # color bar as a legend.
  cbaxes = fig.add_axes([0.25, 0.05, 0.5, 0.02])
  normlegend = mpl.colors.Normalize(vmin=color.min(), vmax=color.max())
  legend = mpl.colorbar.ColorbarBase(cbaxes, norm=normlegend, cmap=colormap, orientation="horizontal",alpha = 0.7,label = "Speed")


  
  return plt.gcf()
  # plt.show()

def driver_gear_shifts(year,gpname,p1):
  # session = ff1.get_session(int(year), str(gpname), 'R')
  session.load()
  p1_name =str(p1)
  desired_driver = p1_name
    
  # Find the desired driver's fastest lap
  laps_of_desired_driver = session.laps.pick_driver(desired_driver)
  fastest_lap_of_driver = laps_of_desired_driver[laps_of_desired_driver['LapTime'] == laps_of_desired_driver['LapTime'].min()]

  # Get telemetry data for the fastest lap of the desired driver
  telemetry_of_driver = fastest_lap_of_driver.get_telemetry()

  # Extract X, Y coordinates and gear data
  x = np.array(telemetry_of_driver['X'].values)
  y = np.array(telemetry_of_driver['Y'].values)
  gear = telemetry_of_driver['nGear'].to_numpy().astype(float)

  # Prepare data for LineCollection
  points = np.array([x, y]).T.reshape(-1, 1, 2)
  segments = np.concatenate([points[:-1], points[1:]], axis=1)

  # Plotting
  cmap = cm.get_cmap('Paired')
  lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N+1), cmap=cmap)
  lc_comp.set_array(gear)
  lc_comp.set_linewidth(4)
  plt.gca().add_collection(lc_comp)
  plt.axis('equal')
  plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

  # Set title
  title = plt.suptitle(
      f"Gear Shift Visualization - {desired_driver}\n"
      f"{session.event['EventName']} {session.event.year}"
  )

  # Add colorbar
  cbar = plt.colorbar(mappable=lc_comp, label="Gear", boundaries=np.arange(1, 10))
  cbar.set_ticks(np.arange(1.5, 9.5))
  cbar.set_ticklabels(np.arange(1, 9))

  # plt.show()
  return plt.gcf()

def team_speeds(year,gpname):
  race = ff1.get_session(int(year),str(gpname), 'R')
  race.load()
  laps = race.laps.pick_quicklaps()
  transformed_laps = laps.copy()
  transformed_laps.loc[:, "LapTime (s)"] = laps["LapTime"].dt.total_seconds()

  # order the team from the fastest (lowest median lap time) tp slower
  team_order = (
      transformed_laps[["Team", "LapTime (s)"]]
      .groupby("Team")
      .median()["LapTime (s)"]
      .sort_values()
      .index
  )
  print(team_order)

  # make a color palette associating team names to hex codes
  team_palette = {team: ff1.plotting.team_color(team) for team in team_order}
  fig, ax = plt.subplots(figsize=(15, 10))
  sns.boxplot(
      data=transformed_laps,
      x="Team",
      y="LapTime (s)",
      order=team_order,
      palette=team_palette,
      whiskerprops=dict(color="white"),
      boxprops=dict(edgecolor="white"),
      medianprops=dict(color="grey"),
      capprops=dict(color="white"),
      legend = False
  )

  plt.title(f"{year} {gpname}")
  plt.grid(visible=False)

  # x-label is redundant
  ax.set(xlabel=None)
  plt.tight_layout()
  return plt.gcf()
  # plt.show()
    
def tyre_strategies(year,gpname):
  # session = ff1.get_session(int(year),str(gpname), 'R')
  # session.load(telemetry=False, weather=False)
  laps = session.laps
  drivers = session.drivers
  drivers = [session.get_driver(driver)["Abbreviation"] for driver in drivers]
  stints = laps[["Driver", "Stint", "Compound", "LapNumber"]]
  stints = stints.groupby(["Driver", "Stint", "Compound"])
  stints = stints.count().reset_index()
  stints = stints.rename(columns={"LapNumber": "StintLength"})
  #print(stints)
  fig, ax = plt.subplots(figsize=(5, 10))

  for driver in drivers:
      driver_stints = stints.loc[stints["Driver"] == driver]

      previous_stint_end =0
      for idx,row in driver_stints.iterrows():
        plt.barh(
                  y= row['Driver'],
                  width=row["StintLength"],
                  left=previous_stint_end,
                  color=ff1.plotting.COMPOUND_COLORS[row["Compound"]],
                  edgecolor="black",
                fill=True
              )

        previous_stint_end += row["StintLength"]

  plt.title(f"Tyre Strategies of all drivers at {gpname}")
  plt.xlabel("Lap Number")
  plt.grid(False)
  # invert the y-axis so drivers that finish higher are closer to the top
  ax.invert_yaxis()
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['left'].set_visible(False)

  plt.tight_layout()
  #plt.savefig("tyre Strategy US 2023")
  #plt.show()
  return plt.gcf()

def position_changes(year,gp_name):
  # session = ff1.get_session(int(year),str(gp_name), 'R')
  # session.load(telemetry=False, weather=False)
  top_10_drivers = session.drivers[:20]
  fig, ax = plt.subplots(figsize=(16, 12))
  for drv in top_10_drivers:
      drv_laps = session.laps.pick_driver(drv)

      abb = drv_laps['Driver'].iloc[0]
      # color = ff1.plotting.driver_color(abb)

      ax.plot(drv_laps['LapNumber'], drv_laps['Position'],
              label=abb)
  ax.set_ylim([15.5, 0.5])
  ax.set_yticks([1, 5, 10,15,20])
  ax.set_xlabel('Lap')
  ax.set_ylabel('Position')
  ax.legend(bbox_to_anchor=(1.0, 1.02))
  plt.tight_layout()
  #plt.savefig("driver standing lap time.png")
  # plt.show()
  return plt.gcf()

def telemetry_comparision(year,gp_name,p1,p2):
  session = ff1.get_session(int(year), str(gp_name), 'R')
  session.load(telemetry = True)
  p1_name =str(p1)
  p2_name = str(p2)
   
  driver_1, driver_2 = p1_name,p2_name
  laps_driver_1 = session.laps.pick_driver(driver_1)
  laps_driver_2 = session.laps.pick_driver(driver_2)

  fastest_driver_1 = laps_driver_1.pick_fastest()
  fastest_driver_2 = laps_driver_2.pick_fastest()

  

  telemetry_driver_1 = fastest_driver_1.get_telemetry()
  telemetry_driver_2 = fastest_driver_2.get_telemetry()
  delta_time, ref_tel, compare_tel = ff1.utils.delta_time(fastest_driver_1, fastest_driver_2)
  team_driver_1 = laps_driver_1['Team'].iloc[0]
  team_driver_2 = laps_driver_2['Team'].iloc[0]

  # Fastf1 has a built-in function for the team colors!
  color_1 = ff1.plotting.team_color(team_driver_1)
  color_2 = ff1.plotting.team_color(team_driver_2)

  plt.rcParams['figure.figsize'] = [20, 15]

  # Our plot will consist of 7 "subplots":
  #     - Delta
  #     - Speed
  #     - Throttle
  #     - Braking
  #     - Gear
  #     - RPM
  #     - DRS
  fig, ax = plt.subplots(7, gridspec_kw={'height_ratios': [1, 3, 2, 1, 1, 2, 1]})

  # Set the title of the plot
  ax[0].title.set_text(f"Telemetry comparison {driver_1} vs. {driver_2}")

  # Subplot 1: The delta
  ax[0].plot(ref_tel['Distance'], delta_time, color=color_1)
  ax[0].axhline(0)
  ax[0].set(ylabel=f"Gap to {driver_2} (s)")

  # Subplot 2: Distance
  ax[1].plot(telemetry_driver_1['Distance'], telemetry_driver_1['Speed'], label=driver_1, color=color_1)
  ax[1].plot(telemetry_driver_2['Distance'], telemetry_driver_2['Speed'], label=driver_2, color=color_2)
  ax[1].set(ylabel='Speed')
  ax[1].legend(loc="lower right")

  # Subplot 3: Throttle
  ax[2].plot(telemetry_driver_1['Distance'], telemetry_driver_1['Throttle'], label=driver_1, color=color_1)
  ax[2].plot(telemetry_driver_2['Distance'], telemetry_driver_2['Throttle'], label=driver_2, color=color_2)
  ax[2].set(ylabel='Throttle')

  # Subplot 4: Brake
  ax[3].plot(telemetry_driver_1['Distance'], telemetry_driver_1['Brake'], label=driver_1, color=color_1)
  ax[3].plot(telemetry_driver_2['Distance'], telemetry_driver_2['Brake'], label=driver_2, color=color_2)
  ax[3].set(ylabel='Brake')

  # Subplot 5: Gear
  ax[4].plot(telemetry_driver_1['Distance'], telemetry_driver_1['nGear'], label=driver_1, color=color_1)
  ax[4].plot(telemetry_driver_2['Distance'], telemetry_driver_2['nGear'], label=driver_2, color=color_2)
  ax[4].set(ylabel='Gear')

  # Subplot 6: RPM
  ax[5].plot(telemetry_driver_1['Distance'], telemetry_driver_1['RPM'], label=driver_1, color=color_1)
  ax[5].plot(telemetry_driver_2['Distance'], telemetry_driver_2['RPM'], label=driver_2, color=color_2)
  ax[5].set(ylabel='RPM')

  # Subplot 7: DRS
  ax[6].plot(telemetry_driver_1['Distance'], telemetry_driver_1['DRS'], label=driver_1, color=color_1)
  ax[6].plot(telemetry_driver_2['Distance'], telemetry_driver_2['DRS'], label=driver_2, color=color_2)
  ax[6].set(ylabel='DRS')
  ax[6].set(xlabel='Lap distance (meters)')

  # Hide x labels and tick labels for top plots and y ticks for right plots.
  for a in ax.flat:
      a.label_outer()
  return plt.gcf()


# Streamlit - Main App
st.title('Formula 1 Race Analysis Dashboard')


with st.sidebar:
    st.title('Selection')
    year = None
    gp_name = None
    p1 = None
    p2 = None
    selected_year = st.selectbox('Select Year',[None] + list(range(2018, 2024)))
    if selected_year is not None:
      year = selected_year
    
      grand_prix_names = list(ff1.get_event_schedule(year)['EventName'])[0:]

      #Dropdown for Grand Prix Name
      selected_gp_name = st.selectbox('Select Grand Prix Name', [None] + grand_prix_names)
      if selected_gp_name is not None:
        gp_name = selected_gp_name
        session = ff1.get_session(year, str(gp_name), 'R')
        session.load(telemetry=False, weather=False)
        driver_name = list(session.results['Abbreviation'])

        p1 = st.selectbox('Select Driver 1', [None] +driver_name)
        p2 = st.selectbox('Select Driver 2', [None] +driver_name)
 

# Visualization w. custom input
if st.button('Visualize Tyre Strategies'):
    fig_tyre = tyre_strategies(year, gp_name)
    st.pyplot(fig_tyre)

if st.button('Lap Time vs Lap Speed'):
    fig_lapscatter = laptimes_scatter(year, gp_name)
    st.pyplot(fig_lapscatter)

if st.button('Visualize Position Changes'):
    fig_position = position_changes(year, gp_name)
    st.pyplot(fig_position)

if st.button('Visualize Telemetry Comparison'):
    fig_telemetry = telemetry_comparision(year, gp_name, p1, p2)
    st.pyplot(fig_telemetry)

if st.button('Visualize Team Speed'):
    fig_telemetry = team_speeds(year, gp_name)
    st.pyplot(fig_telemetry)

if st.button('Speed map of Driver 1'):
  fig_results = driver_speeds(year,gp_name,p1)
  st.pyplot(fig_results)

if st.button('Gear shifts of Driver 1'):
  fig_results = driver_gear_shifts(year,gp_name,p1)
  st.pyplot(fig_results)

if st.button('Visualize Results'):
  fig_results = results(year)
  st.plotly_chart(fig_results)  

# At the bottom of your Streamlit script, add the following lines:

# st.markdown("""---""")  # Adds a horizontal line for separation
# st.markdown("""
# Built by **Harshal Shrimali**  
# Connect with me on [LinkedIn](https://www.linkedin.com/in/harshalshrimali).
# """, unsafe_allow_html=True)
  
