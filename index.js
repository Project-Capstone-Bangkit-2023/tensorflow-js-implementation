const path = require('path')
const tf = require('@tensorflow/tfjs-node')

const main = async () => {

  const [ tourisms, resultUser, rawTourismsArr ] = await Promise.all([dataTourism.toArray(), dataUser.toArray(), rawTourisms.toArray()])

  let newUser = []

  for (let i = 0; i < tourisms.length; i++) {
    const val = Object.values(resultUser[0])
    val[2] = (val[2] - 3) / 2
    newUser.push(val)
  }

  const resultTourism = tourisms.map(d => {
    const val= Object.values(d)
    val[6] = (val[6]-3) / 2
    val[7] = val[6] / 50
    return val
  })

  const model = await tf.loadLayersModel(`file://${path.resolve(__dirname, 'models', 'model.json')}`)
  const inputTf = tf.tensor2d(newUser, [newUser.length, newUser[0].length])
  const inputTf2 = tf.tensor2d(resultTourism, [resultTourism.length, resultTourism[0].length])
  const output = model.predict([inputTf, inputTf2])
  const values = output.dataSync()
  const arr = Array.from(values)
  const sortedArr = Array.from(arr.keys()).sort((a, b) => arr[a] - arr[b])

  for (let i = 0; i < 10; i++) {
    console.log(rawTourismsArr[sortedArr[i]])
  }

  // sortedArr.forEach(idx => {
  //   console.log(tourisms[idx])
  // })

  // console.log(sortedArr)
  // output.print()
}

main();