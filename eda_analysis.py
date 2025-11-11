    def perform_eda(self):
        """Step 3: Exploratory Data Analysis"""
        print("🔍 Performing EDA...")
        
        # Create EDA visualizations
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Delivery Time Distribution
        axes[0,0].hist(self.df['Delivery_Time'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Distribution of Delivery Times', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Delivery Time (Hours)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Impact of Traffic on Delivery Time
        traffic_data = self.df.groupby('Traffic')['Delivery_Time'].mean().sort_values()
        axes[0,1].bar(traffic_data.index, traffic_data.values, color='lightcoral')
        axes[0,1].set_title('Average Delivery Time by Traffic', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Traffic Condition')
        axes[0,1].set_ylabel('Average Delivery Time (Hours)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Impact of Weather on Delivery Time
        weather_data = self.df.groupby('Weather')['Delivery_Time'].mean().sort_values()
        axes[0,2].bar(weather_data.index, weather_data.values, color='lightgreen')
        axes[0,2].set_title('Average Delivery Time by Weather', fontsize=14, fontweight='bold')
        axes[0,2].set_xlabel('Weather Condition')
        axes[0,2].set_ylabel('Average Delivery Time (Hours)')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # 4. Distance vs Delivery Time Scatter Plot
        axes[1,0].scatter(self.df['Distance_km'], self.df['Delivery_Time'], alpha=0.6, color='blue')
        axes[1,0].set_title('Distance vs Delivery Time', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Distance (km)')
        axes[1,0].set_ylabel('Delivery Time (Hours)')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Agent Rating vs Delivery Time
        axes[1,1].scatter(self.df['Agent_Rating'], self.df['Delivery_Time'], alpha=0.6, color='orange')
        axes[1,1].set_title('Agent Rating vs Delivery Time', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Agent Rating')
        axes[1,1].set_ylabel('Delivery Time (Hours)')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Correlation Heatmap
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        im = axes[1,2].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        axes[1,2].set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
        axes[1,2].set_xticks(range(len(correlation_matrix.columns)))
        axes[1,2].set_yticks(range(len(correlation_matrix.columns)))
        axes[1,2].set_xticklabels(correlation_matrix.columns, rotation=90)
        axes[1,2].set_yticklabels(correlation_matrix.columns)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1,2])
        
        plt.tight_layout()
        plt.savefig('eda_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Key insights
        print("\n📈 EDA Key Insights:")
        print(f"• Average Delivery Time: {self.df['Delivery_Time'].mean():.2f} hours")
        print(f"• Max Delivery Time: {self.df['Delivery_Time'].max():.2f} hours")
        print(f"• Min Delivery Time: {self.df['Delivery_Time'].min():.2f} hours")
        print(f"• Correlation (Distance vs Delivery Time): {self.df['Distance_km'].corr(self.df['Delivery_Time']):.3f}")
        print(f"• Correlation (Agent Rating vs Delivery Time): {self.df['Agent_Rating'].corr(self.df['Delivery_Time']):.3f}")
        
        return self.df