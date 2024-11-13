This tool was developed for creating walkable/driveable "turfs" for canvassing based on a custom address list.  It was developed for get-out-the-vote efforts during the Harris/Walz 2024 campaign.  It is designed to consume CSV data downloaded from VoteBuilder/VAN, and creates two outputs: a KML for upload to Google MyMaps, and a set of spreadsheets which are to be printed for use by volunteers.  

There are two relevant scripts:
- turfcutter.py: Data processing, clustering, KML generation, raw XLSX generation
- excel_sheet_cleaner.vbm: Visual Basic Macro which does all the formatting within Excel to go from raw Excel data to a well-formatted spreadsheet which can be directly printed and used by volunteers.

The general workflow is:

1. Within turfcutter.py
- Address validation and lat/lon extraction: GMaps API
- Outlier removal: DBSCAN
- Clustering: Affinity Propogation
- Save lat/lon with address as fields in KML file
- Save cluster addresses and voter info to Excel sheets
2. Manual efforts:
- Upload KML file to Google MyMaps
- Run excel_sheet_cleaner.vbm in Excel to set up for printing
- Dispatch volunteers!


