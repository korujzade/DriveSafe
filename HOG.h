//
// Created by ko on 07/02/16.
//

#ifndef DRIVESAFE_HOG_H
#define DRIVESAFE_HOG_H

using namespace std;

class HOG {
public:
    void generateFeatures(string dir_to_read_bikes, string dir_to_read_bike_annotations,
                          string dir_to_read_front_back_view_bikes, string dir_to_negative_images,
                          string dir_to_xml_files);

};


#endif //DRIVESAFE_HOG_H
