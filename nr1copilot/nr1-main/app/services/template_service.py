
"""
ViralClip Pro v6.0 - Netflix-Level Viral Template Service
Advanced template library with brand kit customization and analytics
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class TemplateCategory(Enum):
    """Template categories for organization"""
    TRENDING = "trending"
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    LIFESTYLE = "lifestyle"
    BUSINESS = "business"
    GAMING = "gaming"
    BEAUTY = "beauty"
    FITNESS = "fitness"
    FOOD = "food"
    TRAVEL = "travel"


class PlatformType(Enum):
    """Supported platforms for template optimization"""
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"
    TWITTER = "twitter"
    SNAPCHAT = "snapchat"
    UNIVERSAL = "universal"


@dataclass
class BrandKit:
    """Brand kit configuration for consistent visual identity"""
    brand_id: str
    name: str
    primary_color: str
    secondary_color: str
    accent_color: str
    background_color: str
    text_color: str
    fonts: Dict[str, str]  # {role: font_name}
    logo_url: Optional[str] = None
    watermark_url: Optional[str] = None
    color_palette: List[str] = field(default_factory=list)
    typography_scale: Dict[str, float] = field(default_factory=dict)
    spacing_scale: Dict[str, float] = field(default_factory=dict)
    border_radius: float = 8.0
    shadow_style: str = "soft"
    animation_style: str = "smooth"
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TemplateElement:
    """Individual template element configuration"""
    element_id: str
    element_type: str  # text, image, video, shape, etc.
    position: Dict[str, float]  # {x, y, width, height}
    style: Dict[str, Any]
    content: Optional[str] = None
    animations: List[Dict[str, Any]] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    layer_order: int = 0


@dataclass
class ViralTemplate:
    """Viral template with comprehensive configuration"""
    template_id: str
    name: str
    description: str
    category: TemplateCategory
    platform: PlatformType
    viral_score: float
    usage_count: int
    elements: List[TemplateElement]
    brand_customizable: bool
    dimensions: Dict[str, int]  # {width, height}
    duration: float
    preview_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    viral_factors: List[str] = field(default_factory=list)
    engagement_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TemplateUsage:
    """Template usage analytics"""
    usage_id: str
    template_id: str
    user_id: str
    brand_kit_id: Optional[str]
    platform: PlatformType
    customizations: Dict[str, Any]
    performance_metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class NetflixLevelTemplateService:
    """Netflix-level viral template service with advanced customization"""

    def __init__(self):
        self.templates: Dict[str, ViralTemplate] = {}
        self.brand_kits: Dict[str, BrandKit] = {}
        self.usage_analytics: List[TemplateUsage] = []
        
        # Initialize with viral templates
        asyncio.create_task(self._initialize_viral_templates())
        
        logger.info("🎨 Netflix-level template service initialized")

    async def _initialize_viral_templates(self):
        """Initialize the viral template library"""
        
        viral_templates = [
            {
                "name": "TikTok Trending Reveal",
                "description": "High-engagement reveal template with trending transitions",
                "category": TemplateCategory.TRENDING,
                "platform": PlatformType.TIKTOK,
                "viral_score": 0.92,
                "viral_factors": ["Quick transitions", "Text reveals", "Hook opening", "Trending audio"],
                "dimensions": {"width": 1080, "height": 1920},
                "duration": 15.0
            },
            {
                "name": "Instagram Story Aesthetic",
                "description": "Minimalist aesthetic template for Instagram Stories",
                "category": TemplateCategory.LIFESTYLE,
                "platform": PlatformType.INSTAGRAM,
                "viral_score": 0.87,
                "viral_factors": ["Clean design", "Text overlays", "Gradient backgrounds"],
                "dimensions": {"width": 1080, "height": 1920},
                "duration": 10.0
            },
            {
                "name": "YouTube Shorts Hook",
                "description": "Attention-grabbing opener for YouTube Shorts",
                "category": TemplateCategory.EDUCATIONAL,
                "platform": PlatformType.YOUTUBE,
                "viral_score": 0.89,
                "viral_factors": ["Strong hook", "Clear typography", "Progress indicators"],
                "dimensions": {"width": 1080, "height": 1920},
                "duration": 30.0
            },
            {
                "name": "Gaming Highlight Montage",
                "description": "High-energy gaming highlight template",
                "category": TemplateCategory.GAMING,
                "platform": PlatformType.UNIVERSAL,
                "viral_score": 0.85,
                "viral_factors": ["Fast cuts", "Beat synchronization", "Impact effects"],
                "dimensions": {"width": 1920, "height": 1080},
                "duration": 20.0
            },
            {
                "name": "Beauty Tutorial Template",
                "description": "Step-by-step beauty tutorial layout",
                "category": TemplateCategory.BEAUTY,
                "platform": PlatformType.INSTAGRAM,
                "viral_score": 0.83,
                "viral_factors": ["Before/after", "Step indicators", "Clean layout"],
                "dimensions": {"width": 1080, "height": 1920},
                "duration": 25.0
            },
            {
                "name": "Food Recipe Quick",
                "description": "Fast-paced recipe template with trending style",
                "category": TemplateCategory.FOOD,
                "platform": PlatformType.TIKTOK,
                "viral_score": 0.91,
                "viral_factors": ["Speed ramping", "Ingredient highlights", "Final reveal"],
                "dimensions": {"width": 1080, "height": 1920},
                "duration": 18.0
            },
            {
                "name": "Fitness Transformation",
                "description": "Motivational fitness progress template",
                "category": TemplateCategory.FITNESS,
                "platform": PlatformType.UNIVERSAL,
                "viral_score": 0.88,
                "viral_factors": ["Progress tracking", "Motivational text", "Time-lapse"],
                "dimensions": {"width": 1080, "height": 1920},
                "duration": 22.0
            },
            {
                "name": "Travel Adventure Story",
                "description": "Epic travel story template with cinematic feel",
                "category": TemplateCategory.TRAVEL,
                "platform": PlatformType.INSTAGRAM,
                "viral_score": 0.86,
                "viral_factors": ["Cinematic transitions", "Location text", "Epic music"],
                "dimensions": {"width": 1080, "height": 1920},
                "duration": 30.0
            },
            {
                "name": "Business Tips Carousel",
                "description": "Professional business tips template",
                "category": TemplateCategory.BUSINESS,
                "platform": PlatformType.UNIVERSAL,
                "viral_score": 0.79,
                "viral_factors": ["Clean design", "Numbered points", "Professional look"],
                "dimensions": {"width": 1080, "height": 1080},
                "duration": 20.0
            },
            {
                "name": "Trending Dance Challenge",
                "description": "High-energy dance challenge template",
                "category": TemplateCategory.ENTERTAINMENT,
                "platform": PlatformType.TIKTOK,
                "viral_score": 0.94,
                "viral_factors": ["Beat sync", "Mirror effects", "Challenge text"],
                "dimensions": {"width": 1080, "height": 1920},
                "duration": 15.0
            },
            {
                "name": "Product Showcase Pro",
                "description": "Professional product showcase template",
                "category": TemplateCategory.BUSINESS,
                "platform": PlatformType.UNIVERSAL,
                "viral_score": 0.81,
                "viral_factors": ["Product focus", "Feature highlights", "CTA elements"],
                "dimensions": {"width": 1080, "height": 1920},
                "duration": 25.0
            },
            {
                "name": "Meme Format Viral",
                "description": "Trending meme format template",
                "category": TemplateCategory.ENTERTAINMENT,
                "platform": PlatformType.UNIVERSAL,
                "viral_score": 0.96,
                "viral_factors": ["Meme format", "Relatable content", "Share-worthy"],
                "dimensions": {"width": 1080, "height": 1080},
                "duration": 8.0
            }
        ]

        for i, template_data in enumerate(viral_templates):
            template_id = f"viral_template_{i+1:03d}"
            
            # Generate template elements
            elements = await self._generate_template_elements(
                template_data["platform"], 
                template_data["category"]
            )
            
            template = ViralTemplate(
                template_id=template_id,
                name=template_data["name"],
                description=template_data["description"],
                category=template_data["category"],
                platform=template_data["platform"],
                viral_score=template_data["viral_score"],
                usage_count=0,
                elements=elements,
                brand_customizable=True,
                dimensions=template_data["dimensions"],
                duration=template_data["duration"],
                preview_url=f"/api/v6/templates/{template_id}/preview",
                thumbnail_url=f"/api/v6/templates/{template_id}/thumbnail",
                tags=template_data.get("tags", []),
                viral_factors=template_data["viral_factors"],
                engagement_metrics={
                    "average_views": 50000 + (template_data["viral_score"] * 100000),
                    "average_engagement": template_data["viral_score"] * 0.1,
                    "share_rate": template_data["viral_score"] * 0.05
                }
            )
            
            self.templates[template_id] = template

        logger.info(f"✅ Initialized {len(self.templates)} viral templates")

    async def _generate_template_elements(
        self, 
        platform: PlatformType, 
        category: TemplateCategory
    ) -> List[TemplateElement]:
        """Generate template elements based on platform and category"""
        
        base_elements = [
            TemplateElement(
                element_id="background",
                element_type="background",
                position={"x": 0, "y": 0, "width": 100, "height": 100},
                style={
                    "type": "gradient",
                    "colors": ["#667eea", "#764ba2"],
                    "direction": "diagonal"
                },
                layer_order=0
            ),
            TemplateElement(
                element_id="main_text",
                element_type="text",
                position={"x": 10, "y": 20, "width": 80, "height": 15},
                style={
                    "font_family": "Inter",
                    "font_weight": "bold",
                    "font_size": 32,
                    "color": "#ffffff",
                    "text_align": "center",
                    "shadow": True
                },
                content="{{main_text}}",
                animations=[
                    {
                        "type": "fade_in",
                        "duration": 0.5,
                        "delay": 0.2
                    }
                ],
                layer_order=10
            )
        ]

        # Platform-specific elements
        if platform == PlatformType.TIKTOK:
            base_elements.append(
                TemplateElement(
                    element_id="trending_badge",
                    element_type="shape",
                    position={"x": 75, "y": 10, "width": 20, "height": 8},
                    style={
                        "shape": "rounded_rect",
                        "color": "#ff6b6b",
                        "border_radius": 15
                    },
                    layer_order=15
                )
            )

        # Category-specific elements
        if category == TemplateCategory.EDUCATIONAL:
            base_elements.append(
                TemplateElement(
                    element_id="step_counter",
                    element_type="text",
                    position={"x": 5, "y": 5, "width": 15, "height": 10},
                    style={
                        "font_family": "Inter",
                        "font_weight": "bold",
                        "font_size": 24,
                        "color": "#ffcc00"
                    },
                    content="{{step_number}}",
                    layer_order=20
                )
            )

        return base_elements

    async def get_viral_templates(
        self,
        category: Optional[TemplateCategory] = None,
        platform: Optional[PlatformType] = None,
        min_viral_score: float = 0.0,
        limit: int = 50
    ) -> List[ViralTemplate]:
        """Get viral templates with filtering options"""
        
        filtered_templates = list(self.templates.values())
        
        # Apply filters
        if category:
            filtered_templates = [t for t in filtered_templates if t.category == category]
        
        if platform:
            filtered_templates = [
                t for t in filtered_templates 
                if t.platform == platform or t.platform == PlatformType.UNIVERSAL
            ]
        
        if min_viral_score > 0:
            filtered_templates = [t for t in filtered_templates if t.viral_score >= min_viral_score]
        
        # Sort by viral score and usage
        filtered_templates.sort(
            key=lambda t: (t.viral_score, t.usage_count), 
            reverse=True
        )
        
        return filtered_templates[:limit]

    async def create_brand_kit(
        self,
        name: str,
        primary_color: str,
        secondary_color: str,
        accent_color: str,
        fonts: Dict[str, str],
        user_id: str
    ) -> BrandKit:
        """Create a new brand kit"""
        
        brand_id = str(uuid.uuid4())
        
        # Generate color palette from primary color
        color_palette = await self._generate_color_palette(primary_color)
        
        # Generate typography scale
        typography_scale = {
            "xs": 12,
            "sm": 14,
            "base": 16,
            "lg": 18,
            "xl": 20,
            "2xl": 24,
            "3xl": 30,
            "4xl": 36
        }
        
        # Generate spacing scale
        spacing_scale = {
            "xs": 4,
            "sm": 8,
            "md": 16,
            "lg": 24,
            "xl": 32,
            "2xl": 48
        }
        
        brand_kit = BrandKit(
            brand_id=brand_id,
            name=name,
            primary_color=primary_color,
            secondary_color=secondary_color,
            accent_color=accent_color,
            background_color="#ffffff",
            text_color="#000000",
            fonts=fonts,
            color_palette=color_palette,
            typography_scale=typography_scale,
            spacing_scale=spacing_scale
        )
        
        self.brand_kits[brand_id] = brand_kit
        
        logger.info(f"🎨 Created brand kit: {name} ({brand_id})")
        return brand_kit

    async def _generate_color_palette(self, primary_color: str) -> List[str]:
        """Generate a color palette from primary color"""
        
        # This would use actual color theory algorithms
        # For demo, return a predefined palette
        palettes = {
            "#667eea": ["#667eea", "#764ba2", "#f093fb", "#f5576c", "#4facfe"],
            "#e50914": ["#e50914", "#ff6b35", "#f7931e", "#ffd700", "#ff4757"],
            "#00ff88": ["#00ff88", "#00ccff", "#667eea", "#764ba2", "#f093fb"]
        }
        
        return palettes.get(primary_color, ["#667eea", "#764ba2", "#f093fb", "#f5576c", "#4facfe"])

    async def customize_template(
        self,
        template_id: str,
        brand_kit_id: Optional[str] = None,
        customizations: Dict[str, Any] = None,
        user_id: str = ""
    ) -> Dict[str, Any]:
        """Customize a template with brand kit and user preferences"""
        
        try:
            template = self.templates.get(template_id)
            if not template:
                raise ValueError(f"Template {template_id} not found")

            brand_kit = None
            if brand_kit_id:
                brand_kit = self.brand_kits.get(brand_kit_id)

            # Create customized template
            customized_template = await self._apply_customizations(
                template, brand_kit, customizations or {}
            )

            # Record usage
            usage = TemplateUsage(
                usage_id=str(uuid.uuid4()),
                template_id=template_id,
                user_id=user_id,
                brand_kit_id=brand_kit_id,
                platform=template.platform,
                customizations=customizations or {},
                performance_metrics={}
            )
            self.usage_analytics.append(usage)

            # Update template usage count
            template.usage_count += 1

            return {
                "success": True,
                "customized_template": customized_template,
                "usage_id": usage.usage_id,
                "viral_score": template.viral_score
            }

        except Exception as e:
            logger.error(f"Template customization failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _apply_customizations(
        self,
        template: ViralTemplate,
        brand_kit: Optional[BrandKit],
        customizations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply brand kit and custom styling to template"""
        
        customized_elements = []
        
        for element in template.elements:
            customized_element = {
                "element_id": element.element_id,
                "element_type": element.element_type,
                "position": element.position.copy(),
                "style": element.style.copy(),
                "content": element.content,
                "animations": element.animations.copy(),
                "layer_order": element.layer_order
            }

            # Apply brand kit styling
            if brand_kit:
                customized_element = await self._apply_brand_kit_styling(
                    customized_element, brand_kit
                )

            # Apply custom overrides
            if element.element_id in customizations:
                element_custom = customizations[element.element_id]
                
                if "style" in element_custom:
                    customized_element["style"].update(element_custom["style"])
                
                if "content" in element_custom:
                    customized_element["content"] = element_custom["content"]
                
                if "position" in element_custom:
                    customized_element["position"].update(element_custom["position"])

            customized_elements.append(customized_element)

        return {
            "template_id": template.template_id,
            "name": template.name,
            "dimensions": template.dimensions,
            "duration": template.duration,
            "elements": customized_elements,
            "viral_factors": template.viral_factors,
            "customization_applied": True
        }

    async def _apply_brand_kit_styling(
        self,
        element: Dict[str, Any],
        brand_kit: BrandKit
    ) -> Dict[str, Any]:
        """Apply brand kit styling to template element"""
        
        element_type = element["element_type"]
        style = element["style"]

        if element_type == "text":
            # Apply brand typography
            if "font_family" in style:
                brand_font = brand_kit.fonts.get("primary", "Inter")
                style["font_family"] = brand_font
            
            # Apply brand colors
            if "color" in style:
                style["color"] = brand_kit.text_color

        elif element_type == "background":
            # Apply brand background
            if style.get("type") == "gradient":
                style["colors"] = [brand_kit.primary_color, brand_kit.secondary_color]
            elif style.get("type") == "solid":
                style["color"] = brand_kit.background_color

        elif element_type == "shape":
            # Apply brand colors to shapes
            if "color" in style:
                style["color"] = brand_kit.accent_color
            
            # Apply brand border radius
            if "border_radius" in style:
                style["border_radius"] = brand_kit.border_radius

        return element

    async def get_template_analytics(
        self,
        template_id: Optional[str] = None,
        user_id: Optional[str] = None,
        time_range: int = 30  # days
    ) -> Dict[str, Any]:
        """Get comprehensive template analytics"""
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=time_range)
        
        # Filter usage data
        usage_data = [
            usage for usage in self.usage_analytics
            if (not template_id or usage.template_id == template_id) and
               (not user_id or usage.user_id == user_id) and
               start_date <= usage.timestamp <= end_date
        ]

        if template_id:
            # Single template analytics
            template = self.templates.get(template_id)
            if not template:
                return {"error": "Template not found"}

            template_usage = [u for u in usage_data if u.template_id == template_id]
            
            return {
                "template_id": template_id,
                "template_name": template.name,
                "viral_score": template.viral_score,
                "total_usage": len(template_usage),
                "platform_breakdown": self._get_platform_breakdown(template_usage),
                "usage_trend": self._get_usage_trend(template_usage, time_range),
                "average_performance": template.engagement_metrics,
                "viral_factors": template.viral_factors
            }
        else:
            # Overall analytics
            return {
                "overview": {
                    "total_templates": len(self.templates),
                    "total_usage": len(usage_data),
                    "most_popular": self._get_most_popular_templates(),
                    "trending_categories": self._get_trending_categories(usage_data)
                },
                "platform_analytics": self._get_platform_analytics(usage_data),
                "viral_score_distribution": self._get_viral_score_distribution(),
                "usage_trends": self._get_overall_usage_trends(usage_data, time_range)
            }

    def _get_platform_breakdown(self, usage_data: List[TemplateUsage]) -> Dict[str, int]:
        """Get platform usage breakdown"""
        breakdown = {}
        for usage in usage_data:
            platform = usage.platform.value
            breakdown[platform] = breakdown.get(platform, 0) + 1
        return breakdown

    def _get_usage_trend(self, usage_data: List[TemplateUsage], days: int) -> List[Dict[str, Any]]:
        """Get usage trend over time"""
        trend = []
        end_date = datetime.utcnow()
        
        for i in range(days):
            date = end_date - timedelta(days=i)
            day_usage = len([
                u for u in usage_data 
                if u.timestamp.date() == date.date()
            ])
            trend.append({
                "date": date.strftime("%Y-%m-%d"),
                "usage_count": day_usage
            })
        
        return list(reversed(trend))

    def _get_most_popular_templates(self) -> List[Dict[str, Any]]:
        """Get most popular templates by usage"""
        templates_by_usage = sorted(
            self.templates.values(),
            key=lambda t: t.usage_count,
            reverse=True
        )
        
        return [
            {
                "template_id": t.template_id,
                "name": t.name,
                "usage_count": t.usage_count,
                "viral_score": t.viral_score
            }
            for t in templates_by_usage[:10]
        ]

    def _get_trending_categories(self, usage_data: List[TemplateUsage]) -> Dict[str, int]:
        """Get trending template categories"""
        category_usage = {}
        for usage in usage_data:
            template = self.templates.get(usage.template_id)
            if template:
                category = template.category.value
                category_usage[category] = category_usage.get(category, 0) + 1
        
        return dict(sorted(category_usage.items(), key=lambda x: x[1], reverse=True))

    def _get_platform_analytics(self, usage_data: List[TemplateUsage]) -> Dict[str, Any]:
        """Get platform-specific analytics"""
        platform_stats = {}
        
        for platform in PlatformType:
            platform_usage = [u for u in usage_data if u.platform == platform]
            platform_stats[platform.value] = {
                "total_usage": len(platform_usage),
                "unique_templates": len(set(u.template_id for u in platform_usage)),
                "average_viral_score": self._calculate_average_viral_score(platform_usage)
            }
        
        return platform_stats

    def _get_viral_score_distribution(self) -> Dict[str, int]:
        """Get viral score distribution across templates"""
        distribution = {
            "90-100": 0,
            "80-89": 0,
            "70-79": 0,
            "60-69": 0,
            "50-59": 0,
            "below-50": 0
        }
        
        for template in self.templates.values():
            score = template.viral_score * 100
            if score >= 90:
                distribution["90-100"] += 1
            elif score >= 80:
                distribution["80-89"] += 1
            elif score >= 70:
                distribution["70-79"] += 1
            elif score >= 60:
                distribution["60-69"] += 1
            elif score >= 50:
                distribution["50-59"] += 1
            else:
                distribution["below-50"] += 1
        
        return distribution

    def _get_overall_usage_trends(self, usage_data: List[TemplateUsage], days: int) -> Dict[str, Any]:
        """Get overall usage trends"""
        recent_usage = len([
            u for u in usage_data 
            if (datetime.utcnow() - u.timestamp).days <= 7
        ])
        
        previous_usage = len([
            u for u in usage_data 
            if 7 < (datetime.utcnow() - u.timestamp).days <= 14
        ])
        
        growth_rate = 0
        if previous_usage > 0:
            growth_rate = ((recent_usage - previous_usage) / previous_usage) * 100
        
        return {
            "recent_week_usage": recent_usage,
            "previous_week_usage": previous_usage,
            "growth_rate": round(growth_rate, 2),
            "peak_day": self._get_peak_usage_day(usage_data)
        }

    def _calculate_average_viral_score(self, usage_data: List[TemplateUsage]) -> float:
        """Calculate average viral score for usage data"""
        if not usage_data:
            return 0.0
        
        total_score = 0
        for usage in usage_data:
            template = self.templates.get(usage.template_id)
            if template:
                total_score += template.viral_score
        
        return round(total_score / len(usage_data), 3)

    def _get_peak_usage_day(self, usage_data: List[TemplateUsage]) -> str:
        """Get the day with peak usage"""
        day_counts = {}
        for usage in usage_data:
            day = usage.timestamp.strftime("%A")
            day_counts[day] = day_counts.get(day, 0) + 1
        
        if not day_counts:
            return "No data"
        
        return max(day_counts.items(), key=lambda x: x[1])[0]

    async def export_template(
        self,
        template_id: str,
        format_type: str = "json",
        include_analytics: bool = False
    ) -> Dict[str, Any]:
        """Export template in various formats"""
        
        try:
            template = self.templates.get(template_id)
            if not template:
                raise ValueError(f"Template {template_id} not found")

            export_data = {
                "template": {
                    "id": template.template_id,
                    "name": template.name,
                    "description": template.description,
                    "category": template.category.value,
                    "platform": template.platform.value,
                    "viral_score": template.viral_score,
                    "dimensions": template.dimensions,
                    "duration": template.duration,
                    "elements": [
                        {
                            "id": elem.element_id,
                            "type": elem.element_type,
                            "position": elem.position,
                            "style": elem.style,
                            "content": elem.content,
                            "animations": elem.animations,
                            "layer_order": elem.layer_order
                        }
                        for elem in template.elements
                    ],
                    "viral_factors": template.viral_factors
                }
            }

            if include_analytics:
                analytics = await self.get_template_analytics(template_id)
                export_data["analytics"] = analytics

            if format_type.lower() == "json":
                return {
                    "success": True,
                    "format": "json",
                    "data": json.dumps(export_data, indent=2)
                }
            else:
                raise ValueError(f"Unsupported format: {format_type}")

        except Exception as e:
            logger.error(f"Template export failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
