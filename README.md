### Project Structure

```bash
.
├── LICENSE
├── README.md
├── data
│   └── mnist.pkl.gz        # MNIST dataset (compressed format)
├── requirements.txt        # Dependencies
└── src
    ├── __pycache__         # Compiled Python cache files
    ├── main.py             # Main script to run the network
    ├── mnist_loader.py     # Data loader for MNIST
    └── network.py          # Implementation of the neural network
```

### Installation

1.  Clone the Repository:

```bash
git clone https://github.com/Rodrigo-Flores/recognize-handwritten-digits.git
```
```bash
cd recognize-handwritten-digits
```

2.  Install Dependencies:
```bash
pip install -r requirements.txt
```


### Usage

To run the project, execute the following command:

```bash
python src/main.py
```

This will train the neural network on the MNIST dataset and evaluate its accuracy in recognizing handwritten digits.

### Dataset

The dataset file, mnist.pkl.gz, is a compressed version of the MNIST dataset and is located in the data directory. The dataset is loaded by the mnist_loader.py module.

### Acknowledgments

This project is inspired by the book Neural Networks and Deep Learning by Michael Nielsen. It demonstrates the basics of neural network training using a simple, fully connected network.

### License

This project is licensed under the MIT License - see the LICENSE file for details.
