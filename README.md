# Fake News Detection web app
### Checks the authenticity of a news piece from URL.

## Usage
Try it here: https://fakenewsdetectorapp.streamlit.app/

## For Developers
To run this project on your local machine, make sure you have Python 3.9 installed on your machine (it is important to have it added in your PATH). Set-up a Python 3.9 virtual environment, where you downloaded the project files, by using:
```
py -3.9 venv .venv
.venv\Scripts\activate
```
Then, install all dependencies by running the following command:
```
pip install -r requirements.txt
```
To run the app on your local machine, type:
```
streamlit run app.py
```
If you wish to train the model on your machine, just run:
```
python model.py
```
This might take a lot of time, depending on your machine, and whether CUDA is available to you.  
  
If you wish to train the model with new dataset, just replace the files in data/ folder.

To evaluate the model you trained, just run:
```
python evaluate.py
```

## Screenshot

## Questions or Suggestions
Feel free to create [issues](https://github.com/tabbybot/fake-news-detector/issues) here as you need.

## Contribute
Contributions to this project are very welcome! Feel free to fork this project, work on it and then make a pull request.

## Authors
- tabbybot (Tabish Perwaiz)

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/tabbybot/fake-news-detector/blob/main/LICENSE) file for details.

## Donate
You can integrate and use these projects in your applications for Free! You can even change the source code and redistribute. However, if you get some profit from this or you just want to encourage me, feel free to


[![Donate](assets/buy_me_a_coffee.jpg)](https://buymeacoffee.com/tabbybot)


Thank you!
