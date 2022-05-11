import requests
import time

f = open("item_list.txt", 'r')

items = f.readlines()

f.close()

total = len(items)
progress = 0

startTime = time.time()

for item in items:
    print(item)

    itemStartTime = time.time()

    item = item.rstrip('\n')
    item = item.strip()

    # item = "MS-NN-00003-00074"

    manifest_url = "https://cudl.lib.cam.ac.uk/iiif/" + item
    r = requests.get(manifest_url)
    json_data = r.json()

    for sequence in json_data['sequences']:

        for canvas in sequence['canvases']:

            for image in canvas['images']:

                iiif_image = image['resource']['@id']

                image_id = iiif_image.rsplit('/', 1)[-1]
               
                

                print(f'Image ID: {image_id}')
                # print("GET DATA", image_id)
                # break
                # # Watermarked
                # download_image = 'https://images.lib.cam.ac.uk/content/images/' + image_id
                # download_image = download_image.replace('.jp2', '.jpg')

                # IIIF
                download_image = 'https://images.lib.cam.ac.uk/iiif/' + image_id + '/full/max/0/default.jpg'

                print(f'URL: {download_image}')

                save_image = image_id.replace('.jp2', '.jpg')
                outdir = 'images_iiif'
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
                save_image = outdir + save_image

                # # Backoff to prevent the IIIF server being overloaded
                # time.sleep(0.1)

                # Exception Handling for invalid requests
                try:
                    # Creating a request object to store the response
                    ImgRequest = requests.get(download_image)
                    # Verifying whether the specified URL exist or not
                    if ImgRequest.status_code == requests.codes.ok:
                        # Opening a file to write bytes from response content
                        # Storing this object as an image file on the hard drive
                        img = open(save_image, "wb")
                        img.write(ImgRequest.content)
                        img.close()

                        print(f'Saved: {save_image}\n')

                    else:
                        print(ImgRequest.status_code)
                except Exception as e:
                    print(str(e))

    progress = progress + 1
    itemExecutionTime = (time.time() - itemStartTime)

    print(f'{progress} items done out of {total} in {itemExecutionTime:.2f} s.\n')

executionTime = (time.time() - startTime)
print(f'ALL DONE in {executionTime:.2f} s.')
