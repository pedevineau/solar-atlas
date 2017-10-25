import os
import urllib2

__author__ = 'marek'


def get_here_map(marker_lat, marker_lon, zoom, marker_size=12):
	# 	example url:
	#
	# http://image.maps.cit.api.here.com/mia/1.6/mapview?app_id=86vKtZUbsaSSRspxBXdk&app_code=JFLmspVWRBe6rRoWrobbdg&w=265&h=200&t=3&c=-33.896637,18.918114&nodot&z=12&poithm=1&poix0=-33.896637,18.918114;ff0000;ff0000;12;.
	#
	# app_id=86vKtZUbsaSSRspxBXdk                     // pristupove kluce, daj niekde do konfiguracie
	# app_code=JFLmspVWRBe6rRoWrobbdg                 // pristupove kluce, daj niekde do konfiguracie
	# w=265                                           // sirka
	# h=200                                           // vyska
	# c=-33.896637,18.918114                          // center: lat,lng
	# t=3                                             // id pre satellite map
	# nodot                                           // zrusi defaultnu zelenu bodku
	# z=12                                            // zoom range: 0-20
	# poithm=1                                        // nejake nastavenie pre markery, uz neviem presne co to znamena
	# poix0=-33.896637,18.918114;ff0000;ff0000;12;.   // marker, pattern: poix{index}=lat,lng;fill color rgb;text color rgb;size(8-30);custom text
	#
	# aby som v markeri zrusil defaultny text tak nastavim fill color rovnako ako text color a ako custom text dam bodku "."
	# mozes vykreslit aj viac markerov poix0, poix1, poix2, atd.. pripadne aj path keby si potreboval
	img_width, img_height = 640, 640/2  # keep ratio 2
	app_id = '86vKtZUbsaSSRspxBXdk'
	app_code = 'JFLmspVWRBe6rRoWrobbdg'
	url = 'http://image.maps.cit.api.here.com/mia/1.6/mapview?app_id=%s&app_code=%s' % (app_id, app_code)
	url += '&w=%s&h=%s&t=3&c=%s,%s' % (img_width, img_height, marker_lat, marker_lon)
	url += '&f=0&ppi=72&nodot&z=%s&poithm=1&poix0=%s,%s;ff0000;ff0000;%s;.' % (zoom, marker_lat, marker_lon, marker_size)
	# print url
	webSource = urllib2.urlopen(url)  # HTTP GET
	image_data = webSource.read()
	return image_data


def get_google_map(marker_lat, marker_lon, zoom, scale=1):
	# see this: https://developers.google.com/maps/documentation/staticmaps/#Imagesizes
	# scale improves resolution of the image, img width and height define aspect ratio of returned image
	# scale is multiplicator of image size: width 600 px x 2 scale = 1200 px of image width - use it only for mobile devices - readable labels
	# zoom range: 0 - 20+
	# image size: 640x640 max in Free mode
	img_width, img_height = 640, 640/2  # keep ratio 2
	url = 'http://maps.googleapis.com/maps/api/staticmap?center=%s,%s&zoom=%s' % (marker_lat, marker_lon, zoom)
	url += '&size=%dx%d' % (img_width, img_height)
	url += '&scale=%d' % scale
	url += '&maptype=hybrid&format=png&markers=size:mediuml|%s,%s' % (marker_lat, marker_lon)
	# print url
	webSource = urllib2.urlopen(url)  # HTTP GET
	image_data = webSource.read()
	return image_data


def get_map(marker_lat, marker_lon, zoom):
	#  here_marker_size and here_zoom are specific for HERE - different from google
	# image width / height ratio is always 2
	image_data = None
	image_vendor = None
	# try use Google first:
	try:
		image_data = get_google_map(marker_lat, marker_lon, zoom)
		image_vendor = 'GOOGLE'
	except Exception as eg:
		print 'Google map failed: %s' % eg
		# try use Here Maps API if Google fails:
		try:
			image_data = get_here_map(marker_lat, marker_lon, zoom)
			image_vendor = 'HERE'
		except Exception as en:
			print 'HERE map failed: %s' % en
	if image_data is not None:
		return (image_data, image_vendor)
	else:
		raise ValueError('Failed to get static map image.')



if __name__ == '__main__':
	marker_lat = 48.145892
	marker_lon = 17.107137
	zoom = 16

	png_file = open(os.path.join(os.path.split(__file__)[0], "/tmp/map_static.png"), 'w')
	image_data, image_vendor = get_map(marker_lat, marker_lon, zoom)
	print image_vendor
	png_file.write(image_data)
	png_file.close()

	png_file_google = open(os.path.join(os.path.split(__file__)[0], "/tmp/map_google.png"), 'w')
	png_file_google.write(get_google_map(marker_lat, marker_lon, zoom))
	png_file_google.close()

	png_file_nokia = open(os.path.join(os.path.split(__file__)[0], "/tmp/map_here.png"), 'w')
	png_file_nokia.write(get_here_map(marker_lat, marker_lon, zoom))
	png_file_nokia.close()
