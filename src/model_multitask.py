import torch
import torch.nn as nn
import timm


class MultiTaskViT(nn.Module):

    def __init__(self):
        super().__init__()

        # ---------------------------------------------
        # ViT Backbone
        # ---------------------------------------------

        self.backbone = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=True,
            num_classes=0
        )

        # Number of features produced by ViT
        feature_dim = self.backbone.num_features

        # ---------------------------------------------
        # Price Head
        # Predict:
        # Open, High, Low, Close
        #
        # Values are learned as normalized price changes
        # ---------------------------------------------

        self.price_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)
        )

        # ---------------------------------------------
        # High Time Head
        # 75 possible 5-minute time slots
        # ---------------------------------------------

        self.high_time_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 75)
        )

        # ---------------------------------------------
        # Low Time Head
        # 75 possible 5-minute time slots
        # ---------------------------------------------

        self.low_time_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 75)
        )

        # ---------------------------------------------
        # Trend Head
        #
        # 0 = Bearish
        # 1 = Bullish
        # ---------------------------------------------

        self.trend_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )


    def forward(self, x):

        # Extract ViT features
        features = self.backbone(x)

        # Multi-task predictions
        price_output = self.price_head(features)

        high_time_output = self.high_time_head(features)

        low_time_output = self.low_time_head(features)

        trend_output = self.trend_head(features)

        return {
            "prices": price_output,
            "high_time": high_time_output,
            "low_time": low_time_output,
            "trend": trend_output
        }
if __name__ == "__main__":

    print("Creating MultiTaskViT...")

    model = MultiTaskViT()

    print("\nModel created successfully!")

    print("ViT feature size:", model.backbone.num_features)

    # Test image
    test_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(test_input)

    print("\nOutput shapes:")

    print("Prices:", output["prices"].shape)
    print("High Time:", output["high_time"].shape)
    print("Low Time:", output["low_time"].shape)
    print("Trend:", output["trend"].shape)