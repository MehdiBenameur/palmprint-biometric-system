using Newtonsoft.Json;

namespace Palmprint.Infrastructure.Ai;

public class GenerateEmbeddingResponse
{
    [JsonProperty("cnn_embedding")]
    public float[] CnnEmbedding { get; set; } = Array.Empty<float>();

    [JsonProperty("triplet_embedding")]
    public float[] TripletEmbedding { get; set; } = Array.Empty<float>();

    [JsonProperty("quality_score")]
    public double QualityScore { get; set; }

    [JsonProperty("model_version")]
    public string ModelVersion { get; set; } = string.Empty;
}