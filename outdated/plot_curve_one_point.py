latitude_= 36.0
extent_latitude = {'start': 36.0, 'end': 36.5}
longitude_ = 125.5
extent_longitude = {'start': 125.5, 'end': 126.0}
dfb_beginning = 13527
dfb_ending = 13590


from nclib2.dataset import DataSet, np

satellite = 'H08LATLON'
suffix_pattern = '__TMON_{YYYY}_{mm}__SDEG05_r{SDEG5_LATITUDE}_c{SDEG5_LONGITUDE}.nc'
selected_channels = ['IR124_2000']
read_dirs = ['/data/model_data_himawari/sat_data_procseg']

chan_patterns = {}
for channel in selected_channels:
    chan_patterns[channel] = satellite + '_' + channel + suffix_pattern
print(chan_patterns)

from nclib2.dataset import DataSet
import matplotlib.pyplot as plt

channels_content = {}
for channel in chan_patterns:
    dataset = DataSet.read(dirs=read_dirs,
                           extent={
                               'latitude': latitude_,
                               'longitude': longitude_,
                               # 'latitude': extent_latitude,
                               # 'longitude': longitude_,
                               # 'dfb': dfb_beginning,
                               'dfb': {'start': dfb_beginning, 'end': dfb_ending},
                           },
                           file_pattern=chan_patterns[channel],
                           variable_name=channel, fill_value=np.nan, interpolation=None, max_processes=0,
                           )
data = dataset['data']
concat_data = [
]
for day in data:
    concat_data.extend(day)

to_plot = []
for d in concat_data:
    if 0 < d < 300:
        to_plot.append(d)
    # else:
    #     to_plot.append(-1)
del concat_data
days = np.linspace(0, dfb_ending - dfb_beginning, len(to_plot))
plt.plot(days, to_plot, 'b-')
axes = plt.gca()
axes.set_ylim([200, 300])
plt.show()