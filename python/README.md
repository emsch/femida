# Femida Backend

To gen an idea, go to 

https://github.com/emsch/femida/blob/master/docker-compose.yaml

There you'll find python entrypoints

```
...
command: run_producer --root=/var/femida_detect --scan-first --host=mongodb
...
command: run_pdf_consumer --root=/var/femida_detect --host=mongodb
...
command: run_answers_consumer --root=/var/femida_detect --host=mongodb --model-path=/model/model.t7
...
command: run_status_updater --root=/var/femida_detect --host=mongodb
```

These 4 binaries is the whole processing pipeline

* `run_producer` waits for events in mongodb and creates tasks to convert a pdf to images
* `run_pdf_consumer` sequentially converts pdfs to images from the queue above and writes images (one pdf = many images)
* `run_answers_consumer` crops images to boxes and runs a neural net for checkmark detection, stores results in mongo (per image)
* `run_status_updater` updates statuses for the batch (per batch)
