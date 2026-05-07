namespace Palmprint.Application.DTOs;

public class IdentifyResponse
{
    public bool Accepted { get; set; }
    public Guid? UserId { get; set; }
    public string? FullName { get; set; }
    public string? ExternalId { get; set; }
    public double SimilarityScore { get; set; }
    public double Threshold { get; set; }
    public List<IdentifyCandidateDto> TopCandidates { get; set; } = new();
    public string Message { get; set; } = string.Empty;
}