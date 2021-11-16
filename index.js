import * as tf from '@tensorflow/tfjs';
import * as tfd from '@tensorflow/tfjs-data';
import {ControllerDataset} from './controller_dataset';
import {maybeRenderDuringTraining, onGameEnd, setUpUI} from './ui';
import * as utils from './utils';

// // knn classifier added
// let classifier = knnClassifier.create();

// predicting 2 classes for left, and right.
const NUM_CLASSES = 2;

// A webcam iterator that generates Tensors from the images from the webcam.
let webcam;

// The dataset object where we will store activations.
const controllerDataset = new ControllerDataset(NUM_CLASSES);

let truncatedMobileNet;
let model;

// Loads mobilenet and returns a model that returns the internal activation
// we'll use as input to our classifier model.
async function loadTruncatedMobileNet() {
  const mobilenet = await tf.loadLayersModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Return a model that outputs an internal activation.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

// When the utils buttons are pressed, read a frame from the webcam and associate
// it with the class label given by the button. up, down, left, right are
// labels 0, 1 respectively.
utils.setExampleHandler(async label => {
  let img = await getImage();

  controllerDataset.addExample(truncatedMobileNet.predict(img), label);

  // Draw the preview thumbnail.
  utils.drawThumb(img, label);
  img.dispose();
})

/**
 * Sets up and trains the classifier.
 */
async function train() {
  if (controllerDataset.xs == null) {
    throw new Error('Add some examples before training!');
  }

  // Creates a 2-layer fully connected model. By creating a separate model,
  // rather than adding layers to the mobilenet model, we "freeze" the weights
  // of the mobilenet model, and only train weights from the new model.
  model = tf.sequential({
    layers: [
      // Flattens the input to a vector so we can use it in a dense layer. While
      // technically a layer, this only performs a reshape (and has no training
      // parameters).
      tf.layers.flatten(
          {inputShape: truncatedMobileNet.outputs[0].shape.slice(1)}),
      // Layer 1.
      tf.layers.dense({
        units: utils.getDenseUnits(),
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      // Layer 2. The number of units of the last layer should correspond
      // to the number of classes we want to predict.
      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });

  // Creates the optimizers which drives training of the model.
  const optimizer = tf.train.adam(utils.getLearningRate());
  // We use categoricalCrossentropy which is the loss function we use for
  // categorical classification which measures the error between our predicted
  // probability distribution over classes (probability that an input is of each
  // class), versus the label (100% probability in the true class)>
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

  // We parameterize batch size as a fraction of the entire dataset because the
  // number of examples that are collected depends on how many examples the user
  // collects. This allows us to have a flexible batch size.
  const batchSize =
      Math.floor(controllerDataset.xs.shape[0] * utils.getBatchSizeFraction());
  if (!(batchSize > 0)) {
    throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }

  // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
  model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs: utils.getEpochs(),
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        utils.trainStatus('Loss: ' + logs.loss.toFixed(5));
      }
    }
  });
}

let isPredicting = false;

async function predict() {
  utils.isPredicting();
  while (isPredicting) {
    // Capture the frame from the webcam.
    const img = await getImage();

    // Make a prediction through mobilenet, getting the internal activation of
    // the mobilenet model, i.e., "embeddings" of the input images.
    const embeddings = truncatedMobileNet.predict(img);

    // Make a prediction through our newly-trained model using the embeddings
    // from mobilenet as input.
    const predictions = model.predict(embeddings);

    // Returns the index with the maximum probability. This number corresponds
    // to the class the model thinks is the most probable given the input.
    const predictedClass = predictions.as1D().argMax();
    const classId = (await predictedClass.data())[0];
    img.dispose();

    utils.predictClass(classId);
    await tf.nextFrame();
  }
  utils.donePredicting();
}

/**
 * Captures a frame from the webcam and normalizes it between -1 and 1.
 * Returns a batched image (1-element batch) of shape [1, w, h, c].
 */
async function getImage() {
  const img = await webcam.capture();
  const processedImg =
      tf.tidy(() => img.expandDims(0).toFloat().div(127).sub(1));
  img.dispose();
  return processedImg;
}

document.getElementById('train').addEventListener('click', async () => {
  utils.trainStatus('Training...');
  await tf.nextFrame();
  await tf.nextFrame();
  isPredicting = false;
  train();
});
document.getElementById('predict').addEventListener('click', () => {
  utils.startPacman();
  isPredicting = true;
  predict();
});

async function init() {
  try {
    webcam = await tfd.webcam(document.getElementById('webcam'));
  } catch (e) {
    console.log(e);
    document.getElementById('no-webcam').style.display = 'block';
  }
  truncatedMobileNet = await loadTruncatedMobileNet();

  utils.init();

  // Warm up the model. This uploads weights to the GPU and compiles the WebGL
  // programs so the first time we collect data from the webcam it will be
  // qutilsck.
  const screenShot = await webcam.capture();
  truncatedMobileNet.predict(screenShot.expandDims(0));
  screenShot.dispose();
}

// Initialize the application.
init();

class PolicyNetwork {
  /**
   * Constructor of PolicyNetwork.
   *
   * @param {number | number[] | tf.LayersModel} hiddenLayerSizes
   *   Can be any of the following
   *   - Size of the hidden layer, as a single number (for a single hidden
   *     layer)
   *   - An Array of numbers (for any number of hidden layers).
   *   - An instance of tf.LayersModel.
   */
  constructor(hiddenLayerSizesOrModel) {
    if (hiddenLayerSizesOrModel instanceof tf.LayersModel) {
      this.policyNet = hiddenLayerSizesOrModel;
    } else {
      this.createPolicyNetwork(hiddenLayerSizesOrModel);
    }
  }

  /**
   * Create the underlying model of this policy network.
   *
   * @param {number | number[]} hiddenLayerSizes Size of the hidden layer, as
   *   a single number (for a single hidden layer) or an Array of numbers (for
   *   any number of hidden layers).
   */
  createPolicyNetwork(hiddenLayerSizes) {
    if (!Array.isArray(hiddenLayerSizes)) {
      hiddenLayerSizes = [hiddenLayerSizes];
    }
    this.policyNet = tf.sequential();
    hiddenLayerSizes.forEach((hiddenLayerSize, i) => {
      this.policyNet.add(tf.layers.dense({
        units: hiddenLayerSize,
        activation: 'elu',
        // `inputShape` is required only for the first layer.
        inputShape: i === 0 ? [4] : undefined
      }));
    });
    // The last layer has only one unit. The single output number will be
    // converted to a probability of selecting the leftward-force action.
    this.policyNet.add(tf.layers.dense({units: 1}));
  }

  /**
   * Train the policy network's model.
   *
   * @param {CartPole} cartPoleSystem The cart-pole system object to use during
   *   training.
   * @param {tf.train.Optimizer} optimizer An instance of TensorFlow.js
   *   Optimizer to use for training.
   * @param {number} discountRate Reward discounting rate: a number between 0
   *   and 1.
   * @param {number} numGames Number of game to play for each model parameter
   *   update.
   * @param {number} maxStepsPerGame Maximum number of steps to perform during
   *   a game. If this number is reached, the game will end immediately.
   * @returns {number[]} The number of steps completed in the `numGames` games
   *   in this round of training.
   */
  async train(
      cartPoleSystem, optimizer, discountRate, numGames, maxStepsPerGame) {
    const allGradients = [];
    const allRewards = [];
    const gameSteps = [];
    onGameEnd(0, numGames);
    for (let i = 0; i < numGames; ++i) {
      // Randomly initialize the state of the cart-pole system at the beginning
      // of every game.
      cartPoleSystem.setRandomState();
      const gameRewards = [];
      const gameGradients = [];
      for (let j = 0; j < maxStepsPerGame; ++j) {
        // For every step of the game, remember gradients of the policy
        // network's weights with respect to the probability of the action
        // choice that lead to the reward.
        const gradients = tf.tidy(() => {
          const inputTensor = cartPoleSystem.getStateTensor();
          return this.getGradientsAndSaveActions(inputTensor).grads;
        });

        this.pushGradients(gameGradients, gradients);
        const action = this.currentActions_[0];
        const isDone = cartPoleSystem.update(action);

        await maybeRenderDuringTraining(cartPoleSystem);

        if (isDone) {
          // When the game ends before max step count is reached, a reward of
          // 0 is given.
          gameRewards.push(0);
          break;
        } else {
          // As long as the game doesn't end, each step leads to a reward of 1.
          // These reward values will later be "discounted", leading to
          // higher reward values for longer-lasting games.
          gameRewards.push(1);
        }
      }
      onGameEnd(i + 1, numGames);
      gameSteps.push(gameRewards.length);
      this.pushGradients(allGradients, gameGradients);
      allRewards.push(gameRewards);
      await tf.nextFrame();
    }

    tf.tidy(() => {
      // The following line does three things:
      // 1. Performs reward discounting, i.e., make recent rewards count more
      //    than rewards from the further past. The effect is that the reward
      //    values from a game with many steps become larger than the values
      //    from a game with fewer steps.
      // 2. Normalize the rewards, i.e., subtract the global mean value of the
      //    rewards and divide the result by the global standard deviation of
      //    the rewards. Together with step 1, this makes the rewards from
      //    long-lasting games positive and rewards from short-lasting
      //    negative.
      // 3. Scale the gradients with the normalized reward values.
      const normalizedRewards =
          discountAndNormalizeRewards(allRewards, discountRate);
      // Add the scaled gradients to the weights of the policy network. This
      // step makes the policy network more likely to make choices that lead
      // to long-lasting games in the future (i.e., the crux of this RL
      // algorithm.)
      optimizer.applyGradients(
          scaleAndAverageGradients(allGradients, normalizedRewards));
    });
    tf.dispose(allGradients);
    return gameSteps;
  }

  getGradientsAndSaveActions(inputTensor) {
    const f = () => tf.tidy(() => {
      const [logits, actions] = this.getLogitsAndActions(inputTensor);
      this.currentActions_ = actions.dataSync();
      const labels =
          tf.sub(1, tf.tensor2d(this.currentActions_, actions.shape));
      return tf.losses.sigmoidCrossEntropy(labels, logits).asScalar();
    });
    return tf.variableGrads(f);
  }

  getCurrentActions() {
    return this.currentActions_;
  }

  /**
   * Get policy-network logits and the action based on state-tensor inputs.
   *
   * @param {tf.Tensor} inputs A tf.Tensor instance of shape `[batchSize, 4]`.
   * @returns {[tf.Tensor, tf.Tensor]}
   *   1. The logits tensor, of shape `[batchSize, 1]`.
   *   2. The actions tensor, of shape `[batchSize, 1]`.
   */
  getLogitsAndActions(inputs) {
    return tf.tidy(() => {
      const logits = this.policyNet.predict(inputs);

      // Get the probability of the leftward action.
      const leftProb = tf.sigmoid(logits);
      // Probabilites of the left and right actions.
      const leftRightProbs = tf.concat([leftProb, tf.sub(1, leftProb)], 1);
      const actions = tf.multinomial(leftRightProbs, 1, null, true);
      return [logits, actions];
    });
  }

  /**
   * Get actions based on a state-tensor input.
   *
   * @param {tf.Tensor} inputs A tf.Tensor instance of shape `[batchSize, 4]`.
   * @param {Float32Array} inputs The actions for the inputs, with length
   *   `batchSize`.
   */
  getActions(inputs) {
    return this.getLogitsAndActions(inputs)[1].dataSync();
  }

  /**
   * Push a new dictionary of gradients into records.
   *
   * @param {{[varName: string]: tf.Tensor[]}} record The record of variable
   *   gradient: a map from variable name to the Array of gradient values for
   *   the variable.
   * @param {{[varName: string]: tf.Tensor}} gradients The new gradients to push
   *   into `record`: a map from variable name to the gradient Tensor.
   */
  pushGradients(record, gradients) {
    for (const key in gradients) {
      if (key in record) {
        record[key].push(gradients[key]);
      } else {
        record[key] = [gradients[key]];
      }
    }
  }
}

// The IndexedDB path where the model of the policy network will be saved.
const MODEL_SAVE_PATH_ = 'indexeddb://cart-pole-v1';

/**
 * A subclass of PolicyNetwork that supports saving and loading.
 */
export class SaveablePolicyNetwork extends PolicyNetwork {
  /**
   * Constructor of SaveablePolicyNetwork
   *
   * @param {number | number[]} hiddenLayerSizesOrModel
   */
  constructor(hiddenLayerSizesOrModel) {
    super(hiddenLayerSizesOrModel);
  }

  /**
   * Save the model to IndexedDB.
   */
  async saveModel() {
    return await this.policyNet.save(MODEL_SAVE_PATH_);
  }

  /**
   * Load the model fom IndexedDB.
   *
   * @returns {SaveablePolicyNetwork} The instance of loaded
   *   `SaveablePolicyNetwork`.
   * @throws {Error} If no model can be found in IndexedDB.
   */
  static async loadModel() {
    const modelsInfo = await tf.io.listModels();
    if (MODEL_SAVE_PATH_ in modelsInfo) {
      console.log(`Loading existing model...`);
      const model = await tf.loadLayersModel(MODEL_SAVE_PATH_);
      console.log(`Loaded model from ${MODEL_SAVE_PATH_}`);
      return new SaveablePolicyNetwork(model);
    } else {
      throw new Error(`Cannot find model at ${MODEL_SAVE_PATH_}.`);
    }
  }

  /**
   * Check the status of locally saved model.
   *
   * @returns If the locally saved model exists, the model info as a JSON
   *   object. Else, `undefined`.
   */
  static async checkStoredModelStatus() {
    const modelsInfo = await tf.io.listModels();
    return modelsInfo[MODEL_SAVE_PATH_];
  }

  /**
   * Remove the locally saved model from IndexedDB.
   */
  async removeModel() {
    return await tf.io.removeModel(MODEL_SAVE_PATH_);
  }

  /**
   * Get the sizes of the hidden layers.
   *
   * @returns {number | number[]} If the model has only one hidden layer,
   *   return the size of the layer as a single number. If the model has
   *   multiple hidden layers, return the sizes as an Array of numbers.
   */
  hiddenLayerSizes() {
    const sizes = [];
    for (let i = 0; i < this.policyNet.layers.length - 1; ++i) {
      sizes.push(this.policyNet.layers[i].units);
    }
    return sizes.length === 1 ? sizes[0] : sizes;
  }
}

/**
 * Discount the reward values.
 *
 * @param {number[]} rewards The reward values to be discounted.
 * @param {number} discountRate Discount rate: a number between 0 and 1, e.g.,
 *   0.95.
 * @returns {tf.Tensor} The discounted reward values as a 1D tf.Tensor.
 */
function discountRewards(rewards, discountRate) {
  const discountedBuffer = tf.buffer([rewards.length]);
  let prev = 0;
  for (let i = rewards.length - 1; i >= 0; --i) {
    const current = discountRate * prev + rewards[i];
    discountedBuffer.set(current, i);
    prev = current;
  }
  return discountedBuffer.toTensor();
}

/**
 * Discount and normalize reward values.
 *
 * This function performs two steps:
 *
 * 1. Discounts the reward values using `discountRate`.
 * 2. Normalize the reward values with the global reward mean and standard
 *    deviation.
 *
 * @param {number[][]} rewardSequences Sequences of reward values.
 * @param {number} discountRate Discount rate: a number between 0 and 1, e.g.,
 *   0.95.
 * @returns {tf.Tensor[]} The discounted and normalize reward values as an
 *   Array of tf.Tensor.
 */
function discountAndNormalizeRewards(rewardSequences, discountRate) {
  return tf.tidy(() => {
    const discounted = [];
    for (const sequence of rewardSequences) {
      discounted.push(discountRewards(sequence, discountRate))
    }
    // Compute the overall mean and stddev.
    const concatenated = tf.concat(discounted);
    const mean = tf.mean(concatenated);
    const std = tf.sqrt(tf.mean(tf.square(concatenated.sub(mean))));
    // Normalize the reward sequences using the mean and std.
    const normalized = discounted.map(rs => rs.sub(mean).div(std));
    return normalized;
  });
}

/**
 * Scale the gradient values using normalized reward values and compute average.
 *
 * The gradient values are scaled by the normalized reward values. Then they
 * are averaged across all games and all steps.
 *
 * @param {{[varName: string]: tf.Tensor[][]}} allGradients A map from variable
 *   name to all the gradient values for the variable across all games and all
 *   steps.
 * @param {tf.Tensor[]} normalizedRewards An Array of normalized reward values
 *   for all the games. Each element of the Array is a 1D tf.Tensor of which
 *   the length equals the number of steps in the game.
 * @returns {{[varName: string]: tf.Tensor}} Scaled and averaged gradients
 *   for the variables.
 */
function scaleAndAverageGradients(allGradients, normalizedRewards) {
  return tf.tidy(() => {
    const gradients = {};
    for (const varName in allGradients) {
      gradients[varName] = tf.tidy(() => {
        // Stack gradients together.
        const varGradients = allGradients[varName].map(
            varGameGradients => tf.stack(varGameGradients));
        // Expand dimensions of reward tensors to prepare for multiplication
        // with broadcasting.
        const expandedDims = [];
        for (let i = 0; i < varGradients[0].rank - 1; ++i) {
          expandedDims.push(1);
        }
        const reshapedNormalizedRewards = normalizedRewards.map(
            rs => rs.reshape(rs.shape.concat(expandedDims)));
        for (let g = 0; g < varGradients.length; ++g) {
          // This mul() call uses broadcasting.
          varGradients[g] = varGradients[g].mul(reshapedNormalizedRewards[g]);
        }
        // Concatenate the scaled gradients together, then average them across
        // all the steps of all the games.
        return tf.mean(tf.concat(varGradients, 0), 0);
      });
    }
    return gradients;
  });
}

setUpUI();
