# Visualization utilities for AutoML Feature Engineering
"""
Visualization module for plotting and analysis.

This module provides the SelectionVisualizer class for creating various
plots related to feature selection, including importance plots, comparison
charts, correlation matrices, and stability analysis.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import warnings

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class SelectionVisualizer:
    """
    Visualization tools for feature selection analysis.

    This class provides methods for creating publication-ready plots
    for feature importance, method comparison, correlation analysis,
    and selection stability.

    Parameters
    ----------
    figsize : tuple, default=(10, 6)
        Default figure size.
    style : str, default='seaborn-v0_8-whitegrid'
        Matplotlib style to use.
    palette : str, default='viridis'
        Color palette for plots.
    dpi : int, default=100
        Resolution for saved figures.

    Examples
    --------
    >>> from automl_fe.visualization import SelectionVisualizer
    >>> from automl_fe.selection import FilterSelector
    >>> import pandas as pd
    >>> from sklearn.datasets import load_iris
    >>> 
    >>> data = load_iris()
    >>> X = pd.DataFrame(data.data, columns=data.feature_names)
    >>> y = pd.Series(data.target)
    >>> 
    >>> selector = FilterSelector(method='mutual_info')
    >>> selector.fit(X, y)
    >>> 
    >>> viz = SelectionVisualizer()
    >>> fig = viz.plot_feature_importance(selector.feature_importances_)
    >>> plt.show()
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 6),
        style: str = 'seaborn-v0_8-whitegrid',
        palette: str = 'viridis',
        dpi: int = 100,
    ):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization. "
                            "Install with: pip install matplotlib")
        
        self.figsize = figsize
        self.style = style
        self.palette = palette
        self.dpi = dpi
        
        # Try to set style, fall back gracefully
        try:
            plt.style.use(style)
        except OSError:
            try:
                plt.style.use('seaborn-whitegrid')
            except OSError:
                pass  # Use default style

    def plot_feature_importance(
        self,
        importances: Dict[str, float],
        top_n: Optional[int] = None,
        title: str = 'Feature Importance',
        xlabel: str = 'Importance Score',
        color: Optional[str] = None,
        ax: Optional[Any] = None,
        horizontal: bool = True,
    ) -> Any:
        """
        Plot feature importance scores.

        Parameters
        ----------
        importances : dict
            Feature names mapped to importance scores.
        top_n : int, optional
            Number of top features to show. If None, shows all.
        title : str, default='Feature Importance'
            Plot title.
        xlabel : str, default='Importance Score'
            Label for x-axis.
        color : str, optional
            Bar color. If None, uses palette.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        horizontal : bool, default=True
            If True, creates horizontal bar chart.

        Returns
        -------
        matplotlib.axes.Axes
            The plot axes.
        """
        # Sort by importance
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        if top_n is not None:
            sorted_imp = sorted_imp[:top_n]
        
        features = [x[0] for x in sorted_imp]
        scores = [x[1] for x in sorted_imp]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        if color is None:
            if HAS_SEABORN:
                colors = sns.color_palette(self.palette, len(features))
            else:
                colors = plt.cm.get_cmap(self.palette)(np.linspace(0, 1, len(features)))
        else:
            colors = color
        
        if horizontal:
            # Reverse for horizontal (top feature at top)
            features = features[::-1]
            scores = scores[::-1]
            if isinstance(colors, list):
                colors = colors[::-1]
            
            ax.barh(features, scores, color=colors)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Feature')
        else:
            ax.bar(features, scores, color=colors)
            ax.set_ylabel(xlabel)
            ax.set_xlabel('Feature')
            plt.xticks(rotation=45, ha='right')
        
        ax.set_title(title)
        plt.tight_layout()
        
        return ax

    def plot_importance_heatmap(
        self,
        importance_dict: Dict[str, Dict[str, float]],
        title: str = 'Feature Importance Heatmap',
        cmap: str = 'YlOrRd',
        annot: bool = True,
        ax: Optional[Any] = None,
    ) -> Any:
        """
        Plot heatmap of feature importances across methods.

        Parameters
        ----------
        importance_dict : dict
            Nested dict: {method_name: {feature: importance}}.
        title : str, default='Feature Importance Heatmap'
            Plot title.
        cmap : str, default='YlOrRd'
            Colormap for heatmap.
        annot : bool, default=True
            Whether to annotate cells with values.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        matplotlib.axes.Axes
            The plot axes.
        """
        if not HAS_SEABORN:
            warnings.warn("Seaborn required for heatmap. Using basic plot.")
            return self.plot_feature_importance(
                list(importance_dict.values())[0]
            )
        
        # Create DataFrame from importance dict
        df = pd.DataFrame(importance_dict).fillna(0)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.heatmap(
            df, annot=annot, cmap=cmap, ax=ax,
            fmt='.3f', cbar_kws={'label': 'Importance'}
        )
        
        ax.set_title(title)
        ax.set_xlabel('Method')
        ax.set_ylabel('Feature')
        plt.tight_layout()
        
        return ax

    def plot_method_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric: str = 'accuracy_mean',
        error_metric: Optional[str] = None,
        title: str = 'Method Comparison',
        ax: Optional[Any] = None,
    ) -> Any:
        """
        Plot comparison of selection methods.

        Parameters
        ----------
        comparison_df : pd.DataFrame
            DataFrame from SelectionComparator.compare().
        metric : str, default='accuracy_mean'
            Column name for the metric to plot.
        error_metric : str, optional
            Column name for error bars (e.g., 'accuracy_std').
        title : str, default='Method Comparison'
            Plot title.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        matplotlib.axes.Axes
            The plot axes.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        methods = comparison_df['method'].tolist()
        scores = comparison_df[metric].tolist()
        
        if error_metric and error_metric in comparison_df.columns:
            errors = comparison_df[error_metric].tolist()
        else:
            errors = None
        
        x = np.arange(len(methods))
        
        if HAS_SEABORN:
            colors = sns.color_palette(self.palette, len(methods))
        else:
            colors = plt.cm.get_cmap(self.palette)(np.linspace(0, 1, len(methods)))
        
        bars = ax.bar(x, scores, color=colors, yerr=errors, capsize=5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.annotate(f'{score:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        return ax

    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        title: str = 'Feature Correlation Matrix',
        cmap: str = 'coolwarm',
        annot: bool = True,
        ax: Optional[Any] = None,
    ) -> Any:
        """
        Plot feature correlation matrix.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with features.
        features : list, optional
            Subset of features to include.
        title : str, default='Feature Correlation Matrix'
            Plot title.
        cmap : str, default='coolwarm'
            Colormap for heatmap.
        annot : bool, default=True
            Whether to annotate cells.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        matplotlib.axes.Axes
            The plot axes.
        """
        if features is not None:
            df = df[features]
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        if HAS_SEABORN:
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(
                corr, mask=mask, cmap=cmap, annot=annot,
                ax=ax, vmin=-1, vmax=1, center=0,
                fmt='.2f', square=True
            )
        else:
            im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr.columns)
            plt.colorbar(im, ax=ax)
        
        ax.set_title(title)
        plt.tight_layout()
        
        return ax

    def plot_stability(
        self,
        stability_scores: Dict[str, float],
        title: str = 'Selection Stability',
        ax: Optional[Any] = None,
    ) -> Any:
        """
        Plot selection stability across methods.

        Parameters
        ----------
        stability_scores : dict
            Method names mapped to stability scores.
        title : str, default='Selection Stability'
            Plot title.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        matplotlib.axes.Axes
            The plot axes.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        methods = list(stability_scores.keys())
        scores = list(stability_scores.values())
        
        if HAS_SEABORN:
            colors = sns.color_palette(self.palette, len(methods))
        else:
            colors = plt.cm.get_cmap(self.palette)(np.linspace(0, 1, len(methods)))
        
        bars = ax.bar(methods, scores, color=colors)
        
        ax.set_ylabel('Stability Score')
        ax.set_xlabel('Method')
        ax.set_title(title)
        ax.set_ylim(0, 1)
        
        # Add threshold line
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax.legend()
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return ax

    def plot_performance_vs_features(
        self,
        n_features: List[int],
        scores: List[float],
        errors: Optional[List[float]] = None,
        title: str = 'Performance vs Number of Features',
        xlabel: str = 'Number of Features',
        ylabel: str = 'Score',
        ax: Optional[Any] = None,
    ) -> Any:
        """
        Plot model performance vs number of features.

        Parameters
        ----------
        n_features : list
            Number of features for each point.
        scores : list
            Performance scores.
        errors : list, optional
            Error bars (standard deviation).
        title : str
            Plot title.
        xlabel : str
            X-axis label.
        ylabel : str
            Y-axis label.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        matplotlib.axes.Axes
            The plot axes.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(n_features, scores, 'o-', markersize=8, linewidth=2)
        
        if errors is not None:
            ax.fill_between(
                n_features,
                np.array(scores) - np.array(errors),
                np.array(scores) + np.array(errors),
                alpha=0.3
            )
        
        # Mark optimal point
        best_idx = np.argmax(scores)
        ax.scatter([n_features[best_idx]], [scores[best_idx]], 
                  color='red', s=200, marker='*', zorder=5,
                  label=f'Optimal: {n_features[best_idx]} features')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return ax

    def plot_feature_overlap(
        self,
        overlap_df: pd.DataFrame,
        title: str = 'Feature Overlap Between Methods',
        cmap: str = 'Blues',
        annot: bool = True,
        ax: Optional[Any] = None,
    ) -> Any:
        """
        Plot feature overlap heatmap.

        Parameters
        ----------
        overlap_df : pd.DataFrame
            Overlap matrix from SelectionComparator.
        title : str
            Plot title.
        cmap : str
            Colormap.
        annot : bool
            Whether to annotate cells.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        matplotlib.axes.Axes
            The plot axes.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        if HAS_SEABORN:
            sns.heatmap(
                overlap_df, annot=annot, cmap=cmap, ax=ax,
                vmin=0, vmax=1, fmt='.2f',
                cbar_kws={'label': 'Jaccard Similarity'}
            )
        else:
            im = ax.imshow(overlap_df.values, cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks(range(len(overlap_df.columns)))
            ax.set_yticks(range(len(overlap_df.index)))
            ax.set_xticklabels(overlap_df.columns, rotation=45, ha='right')
            ax.set_yticklabels(overlap_df.index)
            plt.colorbar(im, ax=ax, label='Jaccard Similarity')
        
        ax.set_title(title)
        plt.tight_layout()
        
        return ax

    def plot_feature_distribution(
        self,
        df: pd.DataFrame,
        feature: str,
        target: Optional[pd.Series] = None,
        title: Optional[str] = None,
        ax: Optional[Any] = None,
    ) -> Any:
        """
        Plot distribution of a single feature.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with features.
        feature : str
            Name of feature to plot.
        target : pd.Series, optional
            Target variable for coloring.
        title : str, optional
            Plot title.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        matplotlib.axes.Axes
            The plot axes.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        if title is None:
            title = f'Distribution of {feature}'
        
        if target is not None and HAS_SEABORN:
            for val in target.unique():
                mask = target == val
                sns.kdeplot(
                    df.loc[mask, feature],
                    ax=ax, label=f'Class {val}',
                    fill=True, alpha=0.3
                )
            ax.legend()
        else:
            ax.hist(df[feature], bins=30, edgecolor='black', alpha=0.7)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Density' if target is not None else 'Count')
        ax.set_title(title)
        
        plt.tight_layout()
        
        return ax

    def plot_selected_vs_unselected(
        self,
        df: pd.DataFrame,
        selected_features: List[str],
        feature_importances: Dict[str, float],
        title: str = 'Selected vs Unselected Features',
        ax: Optional[Any] = None,
    ) -> Any:
        """
        Compare importance of selected vs unselected features.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with features.
        selected_features : list
            List of selected feature names.
        feature_importances : dict
            Feature importance scores.
        title : str
            Plot title.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        matplotlib.axes.Axes
            The plot axes.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        all_features = list(feature_importances.keys())
        selected_scores = [feature_importances[f] for f in all_features if f in selected_features]
        unselected_scores = [feature_importances[f] for f in all_features if f not in selected_features]
        
        if HAS_SEABORN:
            data = pd.DataFrame({
                'Importance': selected_scores + unselected_scores,
                'Status': ['Selected'] * len(selected_scores) + ['Unselected'] * len(unselected_scores)
            })
            sns.boxplot(x='Status', y='Importance', data=data, ax=ax, palette=['green', 'red'])
        else:
            ax.boxplot([selected_scores, unselected_scores], labels=['Selected', 'Unselected'])
            ax.set_ylabel('Importance')
        
        ax.set_title(title)
        plt.tight_layout()
        
        return ax

    def create_report_figure(
        self,
        feature_importances: Dict[str, float],
        selected_features: List[str],
        comparison_df: Optional[pd.DataFrame] = None,
        correlation_df: Optional[pd.DataFrame] = None,
    ) -> Any:
        """
        Create a comprehensive report figure.

        Parameters
        ----------
        feature_importances : dict
            Feature importance scores.
        selected_features : list
            List of selected features.
        comparison_df : pd.DataFrame, optional
            Method comparison results.
        correlation_df : pd.DataFrame, optional
            Feature correlation matrix.

        Returns
        -------
        matplotlib.figure.Figure
            The report figure.
        """
        n_plots = 2
        if comparison_df is not None:
            n_plots += 1
        if correlation_df is not None:
            n_plots += 1
        
        fig, axes = plt.subplots(
            2, 2, figsize=(14, 10)
        )
        axes = axes.flatten()
        
        # Feature importance
        self.plot_feature_importance(
            feature_importances, top_n=15,
            title='Top 15 Feature Importances',
            ax=axes[0]
        )
        
        # Selected vs unselected
        self.plot_selected_vs_unselected(
            pd.DataFrame(),  # Not needed for this plot
            selected_features,
            feature_importances,
            ax=axes[1]
        )
        
        # Method comparison
        if comparison_df is not None:
            primary_metric = [c for c in comparison_df.columns if '_mean' in c][0]
            self.plot_method_comparison(
                comparison_df, metric=primary_metric,
                ax=axes[2]
            )
        else:
            axes[2].text(0.5, 0.5, 'No comparison data', 
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Method Comparison')
        
        # Correlation matrix
        if correlation_df is not None:
            self.plot_correlation_matrix(
                correlation_df, features=selected_features[:10],
                ax=axes[3]
            )
        else:
            axes[3].text(0.5, 0.5, 'No correlation data',
                        ha='center', va='center', transform=axes[3].transAxes)
            axes[3].set_title('Feature Correlation')
        
        plt.suptitle('Feature Selection Report', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        return fig

    def save_figure(
        self,
        fig: Any,
        path: str,
        dpi: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Save figure to file.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save.
        path : str
            Path to save file.
        dpi : int, optional
            Resolution. Uses instance dpi if not specified.
        **kwargs
            Additional arguments for savefig.
        """
        if dpi is None:
            dpi = self.dpi
        
        fig.savefig(path, dpi=dpi, bbox_inches='tight', **kwargs)
        print(f"Figure saved to {path}")
