# femida
Фемида | Система распознавания тестов ЭМШ


# Getting Started

* Pull the repository
```
git clone https://github.com/emsch/femida.git --recursive
```
You need to have permissions to femida-private to obtain models and envfiles. If you don't have such access:

  1. run the following (in the project root)
  ```
  cat Makefile.template > Makefile
  cat envfile.template > envfile
  cat secret.template > secret
  ```
 2. make changes in env files if required
 3. find or train a ocr model and place it to `./model/model.t7`

* run `docker-compose build`
* run `docker-compose up`
