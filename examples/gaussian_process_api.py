import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import time
from ramsey.data import sample_from_sinus_function



from ramsey.models import GP


class Dataset():
  def __init__(self, x, y) -> None:
    self.x = x
    self.y = y

def load_dataset(num_samples, train_split = 1):
    key = jax.random.PRNGKey(23)
    (x, y), f = sample_from_sinus_function(key, batch_size = 1, num_observations=num_samples)

    n_train = int(num_samples * train_split)

    print('  Training Set: %d/%d Samples (%.1f%%)' % (n_train, num_samples, train_split*100))
    print('  Test Set:     %d/%d Samples (%.1f%%)' % (num_samples-n_train, num_samples, (1-train_split)*100))

    x = jnp.squeeze(x)
    y = jnp.squeeze(y)
    f = jnp.squeeze(f)

    idx = jnp.arange(0, num_samples)
    idx = jax.random.permutation(key, idx, independent=True)

    idx_train = idx[:n_train]
    
    x_train = x[idx_train]
    y_train = y[idx_train]

    x_test = jnp.delete(x,idx_train)
    y_test = jnp.delete(y,idx_train)

    train_data = Dataset(x_train,y_train)
    test_data = Dataset(x_test, y_test)


    return train_data, test_data
    


def main():

  print('Load Dataset')
  train_data, test_data = load_dataset(100, train_split = 0.2)

  print('Create GP')
  gp = GP()
  
  print('Start Training')
  start = time.time()
  gp.train(train_data.x, train_data.y)
  end = time.time()
  print('  Training Duration: %.3fs' % (end - start))

  print('Start Prediction')
  start = time.time()
  x_s = jnp.concatenate((train_data.x, test_data.x))
  x_s = jnp.linspace(jnp.min(x_s), jnp.max(x_s), num = 200)
  mu, cov = gp.predict(x_s)
  end = time.time()
  print('  Prediction Duration: %.3fs' % (end - start))
  

  plt.scatter(train_data.x, train_data.y, color='blue', marker='+', label='y_train')
  # plt.scatter(test_data.x, test_data.y, color='green', marker='+', label='y_test')
  plt.plot(x_s, mu, color='orange', label='fit')

  plt.legend()
  plt.grid()
  plt.show(block = True)



if __name__ == "__main__":
  main()