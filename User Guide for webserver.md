# User Guide for Struct2SL: Synthetic Lethality Prediction Webserver

## 1. Introduction
Struct2SL is a web-based tool designed to predict synthetic lethality pairs in human genes. This tool leverages advanced machine learning algorithms and comprehensive data sources to provide valuable insights for cancer research and drug development. This user guide aims to help researchers and clinicians effectively utilize Struct2SL to explore potential synthetic lethality interactions.

## 2. System Requirements
To use Struct2SL, you need:
- A modern web browser (e.g., Google Chrome, Mozilla Firefox, Microsoft Edge, or Safari).
- An active internet connection to access the webserver.

## 3. Accessing Struct2SL
Struct2SL is accessible via a web interface. To start using the tool, navigate to the following URL:
```
http://120.76.218.216:5000
```
Upon accessing the URL, you will be directed to the Struct2SL homepage.

## 4. Navigating the Web Interface

### 4.1 Homepage
The homepage provides an overview of Struct2SL and its capabilities. Key components include:
- **Logo and Title**: Displays the name and subtitle of the tool.
- **Navigation Menu**: Contains links to different sections of the webserver:
  - **Home**: Returns to the homepage.
  - **Search**: Opens the search interface for querying synthetic lethality pairs.
  - **About Us**: Provides information about the developers and contact details.

### 4.2 Search Interface
To query synthetic lethality pairs, click on the "Search" link in the navigation menu or use the search button on the homepage. This will open the search interface, which includes:
- **Search Form**: Contains input fields for entering gene names.
  - **Gene A**: Required field. Enter the name of the first gene.
  - **Gene B**: Optional field. Enter the name of the second gene (if known).
- **Filter Options**: Allows filtering results by source:
  - **ALL**: Displays results from all sources.
  - **SynlethDB**: Displays results from the SynlethDB database.
  - **Struct2SL**: Displays results predicted by the Struct2SL model.
- **Submit Button**: Initiates the search process.

### 4.3 Results Display
After submitting a query, the results will be displayed in the "Results" section. The results are presented in a tabular format, with the following columns:
- **Gene A**: The name of the first gene.
- **Gene B**: The name of the second gene.
- **Prediction Score**: The confidence score of the predicted synthetic lethality interaction.
- **Predicting Relation**: The type of relationship (e.g., "old_SL" for known interactions or "new_SL" for newly predicted interactions).
- **Source**: Indicates whether the result is from SynlethDB or Struct2SL.

### 4.4 Pagination and Filtering
- **Pagination**: If multiple results are returned, the results will be paginated. Use the "Previous" and "Next" buttons to navigate through the pages.
- **Filtering**: Select a filter option (ALL, SynlethDB, or Struct2SL) to refine the displayed results.

## 5. Example Use Case
### Scenario: Identifying Synthetic Lethality Pairs for a Specific Gene
1. Navigate to the Struct2SL homepage.
2. Click on the "Search" link in the navigation menu.
3. In the search form, enter the name of Gene A (e.g., "BRCA1").
4. Leave the Gene B field empty to retrieve all potential synthetic lethality pairs for BRCA1.
5. Select the desired filter option (e.g., "ALL" to view results from both SynlethDB and Struct2SL).
6. Click the "Submit" button.
7. The results will be displayed in the "Results" section. Review the table to identify potential synthetic lethality pairs and their prediction scores.

## 6. Additional Features
### 6.1 About Us
The "About Us" section provides information about the development team, contact details, and the institution associated with Struct2SL. This section is accessible via the navigation menu.

### 6.2 Data Statistics
The homepage includes a section that provides an overview of the data used in Struct2SL. This includes the number of human genes covered and the source of the data (e.g., SynlethDB).

## 7. Troubleshooting
### Common Issues and Solutions
- **No Results Found**: Ensure that the gene names are correctly spelled and formatted. If no results are returned, the queried gene may not have any synthetic lethality pairs in the database.
- **Loading Results**: If the results do not load, check your internet connection or refresh the page.
- **Filtering Issues**: Ensure that the correct filter option is selected. If filtering does not work as expected, try resetting the filter to "ALL" and reapplying the desired option.

## 8. Contact and Support
For further assistance or inquiries, please contact the development team at:
- **Email**: lijunyi@hit.edu.cn
- **Address**: School of Computer Science and Technology, Harbin Institute of Technology (Shenzhen), Shenzhen, Guang Dong 518055, China.

## 9. Conclusion
Struct2SL is a powerful tool for predicting synthetic lethality pairs in human genes. By following this user guide, researchers and clinicians can effectively utilize the webserver to explore potential gene interactions and gain insights for their studies. We encourage users to provide feedback to help us improve the tool and its functionality.

---

**Note**: This user guide is based on the provided files and assumes that the webserver is operational and accessible at the specified URL. If any changes are made to the webserver or its features, please refer to the updated documentation for the most current information.
