import pandas as pd
import h3

# define the bounding box
min_lat = 37.7088
min_lng = -122.5151
max_lat = 37.7853
max_lng = -122.3569

# create a list of H3 hexagons for the bounding box
hexagons = h3.polyfill(
    h3.boundary([(min_lat, min_lng), (min_lat, max_lng),
                 (max_lat, max_lng), (max_lat, min_lng)]),
    7,
    geo_json=False
)

# create a dataframe of the hexagons
df = pd.DataFrame(hexagons, columns=['hexagon'])

# display the dataframe
print(df)
