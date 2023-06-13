const path = require('path')
const tf = require('@tensorflow/tfjs-node')

const dataTourism = tf.data.csv(`file://${path.resolve(__dirname, 'data.csv')}`)
const dataUser = tf.data.csv(`file://${path.resolve(__dirname, 'user.csv')}`)

const main = async () => {
  let resultTourism = await dataTourism.toArray()
  const resultUser = await dataUser.toArray()
  const newUser = []

  for (let i = 0; i < resultTourism.length; i++) {
    newUser.push(Object.values(resultUser[0]))
  }

  resultTourism = resultTourism.map(d => Object.values(d))

  const model = await tf.loadLayersModel(`file://${path.resolve(__dirname, 'models', 'model.json')}`)
  const inputTf = tf.tensor2d(newUser, [newUser.length, newUser[0].length])
  const inputTf2 = tf.tensor2d(resultTourism, [resultTourism.length, resultTourism[0].length])
  const output = model.predict([inputTf, inputTf2])
  output.print()
}

main();