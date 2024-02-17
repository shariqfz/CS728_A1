import matplotlib.pyplot as plt

def plot_history(dataset_name: str, history: dict, freq: int=25):
  # print(f"TEST history: {history}")
  print(f"TEST history_freq: {freq}")
  num_values = len(history[list(history.keys())[0]])

  plt.figure(figsize=(10, 5))
  x = [freq + i*freq for i in range(0, num_values)]
  
  for metric, values in history.items():
      y = values
      plt.plot(x, values, label=metric)
      plt.scatter(x, history[metric])
      plt.annotate(xy=(x[0],y[0]), text='%.2f' % y[0])
      plt.annotate(xy=(x[-1],y[-1]), text='%.2f' % y[-1])
      

  plt.ylabel('Metric')
  plt.xlabel('epochs')
  plt.legend()
  plt.savefig(f'./plots/{dataset_name}_{num_values * freq}_epochs_complEx.png')
  plt.show()