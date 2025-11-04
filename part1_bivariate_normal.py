import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from urllib.request import urlretrieve
import os

class BivariateNormalDistribution:
    """
    Implementation of bivariate normal distribution from scratch
    """
    
    def __init__(self, mu1, mu2, sigma1, sigma2, rho):
        """
        Initialize bivariate normal distribution parameters
        
        Args:
            mu1, mu2: means of X1 and X2
            sigma1, sigma2: standard deviations of X1 and X2
            rho: correlation coefficient
        """
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho
        
        # Calculate covariance matrix
        self.cov12 = rho * sigma1 * sigma2
        self.covariance_matrix = np.array([[sigma1**2, self.cov12],
                                          [self.cov12, sigma2**2]])
        
        # Calculate determinant
        self.det_cov = sigma1**2 * sigma2**2 * (1 - rho**2)
    
    def pdf(self, x1, x2):
        """
        Calculate probability density function value
        Implemented from scratch using the mathematical formula
        """
        # Normalization constant
        norm_const = 1 / (2 * np.pi * self.sigma1 * self.sigma2 * np.sqrt(1 - self.rho**2))
        
        # Standardized variables
        z1 = (x1 - self.mu1) / self.sigma1
        z2 = (x2 - self.mu2) / self.sigma2
        
        # Exponent calculation
        exponent = -1 / (2 * (1 - self.rho**2)) * (z1**2 - 2*self.rho*z1*z2 + z2**2)
        
        return norm_const * np.exp(exponent)
    
    def pdf_vectorized(self, X1, X2):
        """
        Vectorized version for grid calculations
        """
        return self.pdf(X1, X2)

class BivariateNormalAnalysis:
    """
    Complete analysis class for Part 1
    """
    
    def __init__(self):
        self.data = None
        self.bvn_dist = None
        
    def load_iris_dataset(self):
        """
        Load Iris dataset from UCI repository
        """
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        filename = "iris.data"
        
        if not os.path.exists(filename):
            print("Downloading Iris dataset...")
            urlretrieve(url, filename)
        
        # Load data
        column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        self.data = pd.read_csv(filename, names=column_names)
        
        print(f"Loaded {len(self.data)} samples from Iris dataset")
        print(f"Using sepal_length and sepal_width for bivariate analysis")
        
        return self.data
    
    def calculate_parameters(self):
        """
        Calculate bivariate normal parameters from data
        """
        # Use sepal length and width
        x1 = self.data['sepal_length'].values
        x2 = self.data['sepal_width'].values
        
        # Calculate parameters
        mu1 = np.mean(x1)
        mu2 = np.mean(x2)
        sigma1 = np.std(x1, ddof=1)
        sigma2 = np.std(x2, ddof=1)
        
        # Calculate correlation coefficient
        rho = np.corrcoef(x1, x2)[0, 1]
        
        print(f"\nCalculated Parameters:")
        print(f"μ1 (sepal_length): {mu1:.3f}")
        print(f"μ2 (sepal_width): {mu2:.3f}")
        print(f"σ1: {sigma1:.3f}")
        print(f"σ2: {sigma2:.3f}")
        print(f"ρ (correlation): {rho:.3f}")
        
        # Create bivariate normal distribution
        self.bvn_dist = BivariateNormalDistribution(mu1, mu2, sigma1, sigma2, rho)
        
        return mu1, mu2, sigma1, sigma2, rho
    
    def calculate_pdf_values(self):
        """
        Calculate PDF values for each data point
        """
        x1 = self.data['sepal_length'].values
        x2 = self.data['sepal_width'].values
        
        pdf_values = []
        print("\nCalculating PDF values for each data point...")
        
        for i, (x1_val, x2_val) in enumerate(zip(x1, x2)):
            pdf_val = self.bvn_dist.pdf(x1_val, x2_val)
            pdf_values.append(pdf_val)
            
            if i < 5:  # Show first 5 calculations
                print(f"Point {i+1}: ({x1_val:.1f}, {x2_val:.1f}) -> PDF = {pdf_val:.6f}")
        
        self.data['pdf_values'] = pdf_values
        print(f"\nCalculated PDF values for all {len(pdf_values)} data points")
        
        return pdf_values
    
    def create_contour_plot(self):
        """
        Create contour plot visualization
        """
        # Create grid for contour plot
        x1_range = np.linspace(self.data['sepal_length'].min() - 0.5, 
                              self.data['sepal_length'].max() + 0.5, 100)
        x2_range = np.linspace(self.data['sepal_width'].min() - 0.5, 
                              self.data['sepal_width'].max() + 0.5, 100)
        
        X1, X2 = np.meshgrid(x1_range, x2_range)
        Z = self.bvn_dist.pdf_vectorized(X1, X2)
        
        # Create plot
        plt.figure(figsize=(12, 5))
        
        # Contour plot
        plt.subplot(1, 2, 1)
        contour = plt.contour(X1, X2, Z, levels=15, colors='blue', alpha=0.6)
        plt.contourf(X1, X2, Z, levels=15, alpha=0.3, cmap='Blues')
        
        # Plot actual data points colored by species
        species_colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'green', 'Iris-virginica': 'purple'}
        for species in self.data['species'].unique():
            species_data = self.data[self.data['species'] == species]
            plt.scatter(species_data['sepal_length'], species_data['sepal_width'], 
                       c=species_colors.get(species, 'black'), label=species, s=30, alpha=0.7)
        
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Sepal Width (cm)')
        plt.title('Bivariate Normal Distribution - Contour Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.colorbar(label='PDF Value')
        
        # Filled contour plot
        plt.subplot(1, 2, 2)
        plt.contourf(X1, X2, Z, levels=20, cmap='viridis')
        plt.colorbar(label='PDF Value')
        
        # Plot data points
        for species in self.data['species'].unique():
            species_data = self.data[self.data['species'] == species]
            plt.scatter(species_data['sepal_length'], species_data['sepal_width'], 
                       c='white', edgecolors='black', label=species, s=30, alpha=0.8)
        
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Sepal Width (cm)')
        plt.title('Bivariate Normal Distribution - Filled Contours')
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/bivariate_normal_contour.png', dpi=300, bbox_inches='tight')
        print("\nContour plot saved as 'plots/bivariate_normal_contour.png'")
        
        plt.show()
    
    def create_3d_plot(self):
        """
        Create 3D surface plot
        """
        # Create grid for 3D plot
        x1_range = np.linspace(self.data['sepal_length'].min() - 0.5, 
                              self.data['sepal_length'].max() + 0.5, 50)
        x2_range = np.linspace(self.data['sepal_width'].min() - 0.5, 
                              self.data['sepal_width'].max() + 0.5, 50)
        
        X1, X2 = np.meshgrid(x1_range, x2_range)
        Z = self.bvn_dist.pdf_vectorized(X1, X2)
        
        # Create 3D plot
        fig = plt.figure(figsize=(15, 5))
        
        # Surface plot
        ax1 = fig.add_subplot(131, projection='3d')
        surf = ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('Sepal Length (cm)')
        ax1.set_ylabel('Sepal Width (cm)')
        ax1.set_zlabel('PDF Value')
        ax1.set_title('3D Surface Plot')
        fig.colorbar(surf, ax=ax1, shrink=0.5)
        
        # Wireframe plot
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot_wireframe(X1, X2, Z, alpha=0.6, color='blue')
        ax2.set_xlabel('Sepal Length (cm)')
        ax2.set_ylabel('Sepal Width (cm)')
        ax2.set_zlabel('PDF Value')
        ax2.set_title('3D Wireframe Plot')
        
        # Contour plot with projection
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.contour(X1, X2, Z, levels=15, cmap='plasma')
        
        # Add data points as vertical lines
        for i, row in self.data.head(20).iterrows():  # Show first 20 points
            x1_val, x2_val = row['sepal_length'], row['sepal_width']
            pdf_val = self.bvn_dist.pdf(x1_val, x2_val)
            ax3.plot([x1_val, x1_val], [x2_val, x2_val], [0, pdf_val], 'r-', alpha=0.6)
            ax3.scatter([x1_val], [x2_val], [pdf_val], c='red', s=20)
        
        ax3.set_xlabel('Sepal Length (cm)')
        ax3.set_ylabel('Sepal Width (cm)')
        ax3.set_zlabel('PDF Value')
        ax3.set_title('3D Contour with Data Points')
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig('plots/bivariate_normal_3d.png', dpi=300, bbox_inches='tight')
        print("3D plot saved as 'plots/bivariate_normal_3d.png'")
        
        plt.show()
    
    def run_complete_analysis(self):
        """
        Run the complete Part 1 analysis
        """
        print("Starting Part 1: Bivariate Normal Distribution Analysis")
        print("=" * 60)
        
        # Load data
        self.load_iris_dataset()
        
        # Calculate parameters
        self.calculate_parameters()
        
        # Calculate PDF values
        self.calculate_pdf_values()
        
        # Create visualizations
        self.create_contour_plot()
        self.create_3d_plot()
        
        print("\nPart 1 Analysis Complete!")
        print("Generated files:")
        print("- plots/bivariate_normal_contour.png")
        print("- plots/bivariate_normal_3d.png")

if __name__ == "__main__":
    analysis = BivariateNormalAnalysis()
    analysis.run_complete_analysis()