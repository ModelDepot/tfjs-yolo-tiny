const path = require('path');
const webpack = require('webpack');

module.exports = {
  context: path.resolve(__dirname, 'src'),
  entry: {
    index: ['./index.js'],
  },
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: '[name].bundle.js',
    library: 'remoteRequire',
    libraryTarget: 'umd',
  },
  module: {
    rules: [
      {
        test: /\.js$/i,
        exclude: [/node_modules/],
        use: [{
          loader: 'babel-loader',
          options: { presets: ['env'] },
        }],
      },
    ],
  },
  externals: {
    '@tensorflow/tfjs': '@tensorflow/tfjs',
  },
};
